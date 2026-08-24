import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import BatchSampler, DataLoader

ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    str(ROOT_DIR / ".torch_extensions"),
)
sys.path.insert(0, str(ROOT_DIR / "src"))

from dust3r.datasets import *  # noqa: F401,F403,E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.surfel_render import Gaussian2DRenderingLoss, render_surfels  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402


class DistributedFixedViewBatchSampler(BatchSampler):
    """Shard BaseMultiViewDataset tuple indices across DDP ranks."""

    def __init__(
        self,
        dataset,
        batch_size,
        rank,
        world_size,
        seed=788,
        drop_last=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = len(dataset)
        self.num_aspect_ratios = len(dataset._resolutions)
        self.num_views = dataset.num_views

        if self.drop_last:
            self.total_size = (
                self.num_samples // (self.world_size * self.batch_size)
            ) * self.world_size * self.batch_size
        else:
            self.total_size = int(
                np.ceil(self.num_samples / (self.world_size * self.batch_size))
            ) * self.world_size * self.batch_size
        self.rank_samples = self.total_size // self.world_size
        self.rank_batches = self.rank_samples // self.batch_size
        if self.rank_batches <= 0:
            raise ValueError(
                "Dataset is too small for the requested DDP world size and "
                f"batch size: samples={self.num_samples}, "
                f"world_size={self.world_size}, batch_size={self.batch_size}"
            )

    def __len__(self):
        return self.rank_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.permutation(self.num_samples).tolist()

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding = self.total_size - len(indices)
            if padding > 0:
                indices += indices[:padding]

        rank_indices = indices[self.rank : self.total_size : self.world_size]
        rank_indices = rank_indices[: self.rank_batches * self.batch_size]

        for batch_id in range(self.rank_batches):
            start = batch_id * self.batch_size
            sample_ids = rank_indices[start : start + self.batch_size]
            aspect_ratio_id = int(rng.integers(self.num_aspect_ratios))
            yield [
                (sample_id, aspect_ratio_id, self.num_views)
                for sample_id in sample_ids
            ]


def parse_args():
    parser = argparse.ArgumentParser(
        "DDP train the frozen TTT3R DPT-2DGS head"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--gaussian_2d_feature_dim", type=int, default=64)
    parser.add_argument("--gaussian_2d_scale_min", type=float, default=1e-4)
    parser.add_argument("--gaussian_2d_scale_init", type=float, default=1e-2)
    parser.add_argument("--gaussian_2d_opacity_init", type=float, default=0.5)
    parser.add_argument("--render_stride", type=int, default=1)
    parser.add_argument("--self_weight", type=float, default=1.0)
    parser.add_argument("--merged_weight", type=float, default=1.0)
    parser.add_argument("--rgb_weight", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=0.02)
    parser.add_argument("--normal_weight", type=float, default=0.1)
    parser.add_argument(
        "--depth_target_source",
        choices=("auto", "ground_truth", "predicted"),
        default="ground_truth",
        help="Depth source used only to derive the normal target.",
    )
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--save_every_steps", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--dist_timeout_minutes", type=int, default=120)
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=0,
        help="MAX_JOBS used by gsplat CUDA extension build; <=0 leaves env unchanged.",
    )
    return parser.parse_args()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed(args):
    if "RANK" not in os.environ:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for 2DGS rasterization")
        return torch.device(args.device)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP 2DGS rasterization")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=args.dist_backend,
        timeout=timedelta(minutes=args.dist_timeout_minutes),
    )
    return torch.device("cuda", local_rank)


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


def checkpoint_config(args):
    return {
        "version": 4,
        "loss_name": "mse_lpips_normal_2d",
        "gaussian_2d_feature_dim": args.gaussian_2d_feature_dim,
        "gaussian_2d_scale_min": args.gaussian_2d_scale_min,
        "gaussian_2d_scale_init": args.gaussian_2d_scale_init,
        "gaussian_2d_scale_max": None,
        "gaussian_2d_opacity_init": args.gaussian_2d_opacity_init,
        "render_stride": args.render_stride,
        "self_weight": args.self_weight,
        "merged_weight": args.merged_weight,
        "rgb_weight": args.rgb_weight,
        "lpips_weight": args.lpips_weight,
        "normal_weight": args.normal_weight,
        "depth_target_source": args.depth_target_source,
    }


def resume_config_changes(saved_config, args):
    current_config = checkpoint_config(args)
    saved_config = saved_config or {}
    strict_keys = {
        "gaussian_2d_feature_dim",
        "gaussian_2d_scale_min",
        "gaussian_2d_scale_init",
        "gaussian_2d_opacity_init",
    }
    mismatches = {
        key: (saved_config.get(key), current_config.get(key))
        for key in strict_keys
        if saved_config.get(key) != current_config.get(key)
    }
    if mismatches:
        raise ValueError(
            "Resume checkpoint 2DGS head configuration does not match: "
            f"{mismatches}"
        )
    return {
        key: (saved_config.get(key), value)
        for key, value in current_config.items()
        if key not in strict_keys and saved_config.get(key) != value
    }


def save_checkpoint(path, model, optimizer, epoch, step, args):
    torch.save(
        {
            "gaussian_2d_head": (
                model.downstream_head.dpt_gaussian_2d.state_dict()
            ),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": checkpoint_config(args),
        },
        path,
    )


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def reduce_loss_and_details(loss, details, device):
    if not is_dist():
        return float(loss.detach()), {
            key: float(value.detach()) if torch.is_tensor(value) else float(value)
            for key, value in details.items()
        }

    keys = sorted(details)
    values = [loss.detach()]
    for key in keys:
        value = details[key]
        if not torch.is_tensor(value):
            value = torch.tensor(float(value), device=device)
        else:
            value = value.detach().to(device)
        values.append(value.float())
    packed = torch.stack([value.float() for value in values])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= get_world_size()
    return float(packed[0].cpu()), {
        key: float(packed[index + 1].cpu())
        for index, key in enumerate(keys)
    }


def build_ddp_loader(dataset, args):
    sampler = DistributedFixedViewBatchSampler(
        dataset,
        batch_size=args.batch_size,
        rank=get_rank(),
        world_size=get_world_size(),
        seed=args.seed + 788,
        drop_last=True,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _warmup_gsplat_extension_once(device):
    means = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    colors = torch.tensor([[0.5, 0.5, 0.5]], device=device)
    opacities = torch.tensor([[0.5]], device=device)
    scales = torch.tensor([[0.01, 0.01]], device=device)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    intrinsics = torch.tensor(
        [[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    with torch.no_grad():
        render_surfels(
            means,
            colors,
            opacities,
            scales,
            rotations,
            intrinsics,
            height=16,
            width=16,
        )
    torch.cuda.synchronize(device)


def warmup_gsplat_extension(device):
    if device.type != "cuda":
        return
    if is_dist():
        if is_main_process():
            print_main("Warming up 2DGS CUDA extension on rank0...", flush=True)
            _warmup_gsplat_extension_once(device)
        dist.barrier()
        if not is_main_process():
            _warmup_gsplat_extension_once(device)
        dist.barrier()
    else:
        _warmup_gsplat_extension_once(device)


def main():
    args = parse_args()
    if args.max_jobs > 0:
        os.environ["MAX_JOBS"] = str(args.max_jobs)
    device = setup_distributed(args)
    rank = get_rank()
    world_size = get_world_size()

    try:
        if args.resume and not Path(args.resume).is_file():
            raise FileNotFoundError(
                f"--resume checkpoint does not exist: {args.resume}. "
                "Remove --resume to start a fresh run, or point --resume to an "
                "existing 2d-gaussian-head-*.pth checkpoint."
            )
        if args.render_stride > 2:
            print_main(
                "WARNING: render_stride > 2 saves memory but removes substantial "
                "texture and normal detail; use it only for smoke tests."
            )
        print_main(
            f"torch_extensions_dir={os.environ['TORCH_EXTENSIONS_DIR']}",
            f"max_jobs={os.environ.get('MAX_JOBS', '<unset>')}",
            flush=True,
        )
        warmup_gsplat_extension(device)

        output_dir = Path(args.output_dir)
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
        if is_dist():
            dist.barrier()

        last_checkpoint = output_dir / "2d-gaussian-head-last.pth"
        if last_checkpoint.exists() and not args.resume:
            raise FileExistsError(
                f"{last_checkpoint} exists; use a new output directory or --resume"
            )
        if is_main_process():
            with (output_dir / "args.json").open("w") as file:
                json.dump(vars(args), file, indent=2)
        if is_dist():
            dist.barrier()

        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)

        print_main(
            "DPT-2DGS DDP setup: "
            f"world_size={world_size} per_gpu_batch={args.batch_size} "
            f"global_batch={world_size * args.batch_size * args.accum_iter} "
            f"scale=softplus(raw)+{args.gaussian_2d_scale_min} "
            f"scale_init={args.gaussian_2d_scale_init} "
            f"opacity={args.gaussian_2d_opacity_init} "
            f"render_stride={args.render_stride} "
            f"loss=self*{args.self_weight}+merged*{args.merged_weight} "
            "render=rgb_mse+lpips+normal "
            f"rgb_weight={args.rgb_weight} "
            f"lpips_weight={args.lpips_weight} "
            f"normal_weight={args.normal_weight} "
            "intrinsics=predicted "
            f"normal_target_depth={args.depth_target_source}"
        )

        model = ARCroco3DStereo.from_pretrained(
            args.model_path,
            gaussian_2d_head=True,
            gaussian_2d_feature_dim=args.gaussian_2d_feature_dim,
            gaussian_2d_scale_min=args.gaussian_2d_scale_min,
            gaussian_2d_scale_init=args.gaussian_2d_scale_init,
            gaussian_2d_scale_max=None,
            gaussian_2d_opacity_init=args.gaussian_2d_opacity_init,
        )
        model.freeze_non_gaussian_2d_parameters()
        model.to(device)
        model.eval()
        model.downstream_head.dpt_gaussian_2d.train()

        ddp_model = (
            DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            if is_dist()
            else model
        )
        raw_model = ddp_model.module if is_dist() else ddp_model

        optimizer = torch.optim.AdamW(
            raw_model.gaussian_2d_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        criterion = Gaussian2DRenderingLoss(
            render_stride=args.render_stride,
            rgb_weight=args.rgb_weight,
            lpips_weight=args.lpips_weight,
            normal_weight=args.normal_weight,
            self_weight=args.self_weight,
            merged_weight=args.merged_weight,
            depth_target_source=args.depth_target_source,
        ).to(device)

        dataset = eval(args.train_dataset)
        loader = build_ddp_loader(dataset, args)
        loader_steps = len(loader)
        epoch_steps = (
            min(loader_steps, args.steps_per_epoch)
            if args.steps_per_epoch
            else loader_steps
        )
        print_main(
            f"dataset_samples={len(dataset)} loader_steps_per_rank={loader_steps} "
            f"steps_per_epoch={epoch_steps} "
            f"planned_optimizer_steps={epoch_steps * args.epochs // args.accum_iter}"
        )

        start_epoch = 0
        global_step = 0
        if args.resume:
            checkpoint = torch.load(
                args.resume, map_location="cpu", weights_only=False
            )
            changed_config = resume_config_changes(checkpoint.get("config"), args)
            if changed_config:
                print_main(
                    "Resume with updated training/render config: "
                    f"{changed_config}",
                    flush=True,
                )
            raw_model.downstream_head.dpt_gaussian_2d.load_state_dict(
                checkpoint["gaussian_2d_head"]
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["step"]
        if is_dist():
            dist.barrier()

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = not args.no_amp
        for epoch in range(start_epoch, args.epochs):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            loader.batch_sampler.set_epoch(epoch)

            for batch_step, batch in enumerate(loader):
                if args.steps_per_epoch and batch_step >= args.steps_per_epoch:
                    break

                batch = todevice(batch, device, non_blocking=True)
                should_step = (batch_step + 1) % args.accum_iter == 0
                sync_context = (
                    nullcontext()
                    if should_step or not is_dist()
                    else ddp_model.no_sync()
                )

                with sync_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        output = ddp_model(batch)
                        loss, details = criterion(output.views, output.ress)
                        scaled_loss = loss / args.accum_iter
                    scaled_loss.backward()

                if not should_step:
                    continue

                clip_grad_norm_(list(raw_model.gaussian_2d_parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss, avg_details = reduce_loss_and_details(
                        loss, details, device
                    )
                    if is_main_process():
                        detail_text = " ".join(
                            f"{key}={value:.4f}"
                            for key, value in avg_details.items()
                        )
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={avg_loss:.4f} {detail_text}",
                            flush=True,
                        )
                if (
                    is_main_process()
                    and args.save_every_steps
                    and global_step % args.save_every_steps == 0
                ):
                    save_checkpoint(
                        last_checkpoint,
                        raw_model,
                        optimizer,
                        epoch,
                        global_step,
                        args,
                    )

            if is_main_process() and (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"2d-gaussian-head-{epoch + 1:04d}.pth",
                    raw_model,
                    optimizer,
                    epoch,
                    global_step,
                    args,
                )
            if is_main_process():
                save_checkpoint(
                    last_checkpoint,
                    raw_model,
                    optimizer,
                    epoch,
                    global_step,
                    args,
                )
            if is_dist():
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    record(main)()
