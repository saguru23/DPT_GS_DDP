import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dust3r.datasets import *  # noqa: F401,F403,E402
from dust3r.datasets import get_data_loader  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.surfel_render import render_prediction_view, sample_surfels  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Render a DPT-2DGS checkpoint")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gaussian_checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--export_ply", action="store_true")
    parser.add_argument("--ply_stride", type=int, default=1)
    parser.add_argument(
        "--ply_format",
        choices=("pointcloud", "supersplat"),
        default="supersplat",
        help="PLY schema to export. supersplat writes Graphdeco/3DGS fields.",
    )
    parser.add_argument(
        "--ply_scale_z",
        type=float,
        default=0.0,
        help=(
            "Fixed z scale for supersplat export. <=0 uses a thin automatic "
            "value from the predicted 2D scales."
        ),
    )
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="Only run the model/exporters; do not call gsplat rendering.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def save_image(path, image):
    image = image.detach().float().cpu().numpy()
    imageio.imwrite(path, (np.clip(image, 0, 1) * 255).astype(np.uint8))


def save_depth(path, depth):
    depth = depth.detach().float()
    valid = torch.isfinite(depth) & (depth > 0)
    if valid.any():
        lo = torch.quantile(depth[valid], 0.02)
        hi = torch.quantile(depth[valid], 0.98)
        depth = (depth - lo) / (hi - lo).clamp_min(1e-6)
    else:
        depth = torch.zeros_like(depth)
    save_image(path, depth.clamp(0, 1).unsqueeze(-1).expand(-1, -1, 3))


def _scales_2d_to_3d(scales, scale_z):
    if scale_z > 0:
        z = np.full((len(scales), 1), scale_z, dtype=np.float32)
    else:
        z = np.minimum(scales[:, :1], scales[:, 1:2]) * 0.05
        z = np.clip(z, 1e-6, None)
    return np.concatenate([scales, z], axis=1)


def save_surfel_ply(
    path,
    means,
    colors,
    opacities,
    scales,
    rotations,
    ply_format="pointcloud",
    scale_z=0.0,
):
    means = means.detach().float().cpu().numpy()
    colors = colors.detach().float().cpu().numpy()
    opacities = opacities.detach().float().cpu().numpy().reshape(-1)
    scales = scales.detach().float().cpu().numpy()
    rotations = rotations.detach().float().cpu().numpy()

    scales3d = _scales_2d_to_3d(scales, scale_z)
    if ply_format == "supersplat":
        save_supersplat_ply(path, means, colors, opacities, scales3d, rotations)
        return

    save_pointcloud_ply(path, means, colors, opacities, scales, rotations, scales3d)


def save_pointcloud_ply(
    path,
    means,
    colors,
    opacities,
    scales2d,
    rotations,
    scales3d,
):
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    with Path(path).open("w") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(means)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("property float opacity\n")
        file.write("property float scale_0\n")
        file.write("property float scale_1\n")
        file.write("property float scale_2\n")
        file.write("property float scale_2d_0\n")
        file.write("property float scale_2d_1\n")
        file.write("property float rot_0\n")
        file.write("property float rot_1\n")
        file.write("property float rot_2\n")
        file.write("property float rot_3\n")
        file.write("end_header\n")
        for xyz, color, opacity, scale2d, scale3d, rotation in zip(
            means, rgb, opacities, scales2d, scales3d, rotations
        ):
            file.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{opacity:.8f} "
                f"{scale3d[0]:.8f} {scale3d[1]:.8f} {scale3d[2]:.8f} "
                f"{scale2d[0]:.8f} {scale2d[1]:.8f} "
                f"{rotation[0]:.8f} {rotation[1]:.8f} "
                f"{rotation[2]:.8f} {rotation[3]:.8f}\n"
            )


def save_supersplat_ply(path, means, colors, opacities, scales, rotations):
    sh_c0 = 0.28209479177387814
    colors = np.clip(colors, 0.0, 1.0)
    opacities = np.clip(opacities, 1e-6, 0.95)
    scales = np.clip(scales, 1e-8, None)
    rotation_norm = np.linalg.norm(rotations, axis=-1, keepdims=True)
    rotations = rotations / np.clip(rotation_norm, 1e-8, None)

    sh_dc = (colors - 0.5) / sh_c0
    sh_rest = np.zeros((len(means), 45), dtype=np.float32)
    opacity_logits = np.log(opacities / (1.0 - opacities))
    log_scales = np.log(scales)

    field_names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{index}" for index in range(3)]
        + [f"f_rest_{index}" for index in range(45)]
        + ["opacity"]
        + [f"scale_{index}" for index in range(3)]
        + [f"rot_{index}" for index in range(4)]
    )
    values = np.concatenate(
        [
            means.astype(np.float32),
            np.zeros_like(means, dtype=np.float32),
            sh_dc.astype(np.float32),
            sh_rest,
            opacity_logits[:, None].astype(np.float32),
            log_scales.astype(np.float32),
            rotations.astype(np.float32),
        ],
        axis=1,
    )

    with Path(path).open("wb") as file:
        file.write(b"ply\n")
        file.write(b"format binary_little_endian 1.0\n")
        file.write(f"element vertex {len(means)}\n".encode("ascii"))
        for field_name in field_names:
            file.write(f"property float {field_name}\n".encode("ascii"))
        file.write(b"end_header\n")
        values.astype("<f4", copy=False).tofile(file)


def main():
    args = parse_args()
    if args.skip_render and not args.export_ply:
        raise ValueError("--skip_render is only useful together with --export_ply")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Use --device cpu with --skip_render "
            "--export_ply, or run rendering in a CUDA environment."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(
        args.gaussian_checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model = ARCroco3DStereo.from_pretrained(
        args.model_path,
        gaussian_2d_head=True,
        gaussian_2d_feature_dim=config["gaussian_2d_feature_dim"],
        gaussian_2d_scale_min=config["gaussian_2d_scale_min"],
        gaussian_2d_scale_init=config["gaussian_2d_scale_init"],
        gaussian_2d_scale_max=config["gaussian_2d_scale_max"],
        gaussian_2d_opacity_init=config["gaussian_2d_opacity_init"],
    )
    model.downstream_head.dpt_gaussian_2d.load_state_dict(
        checkpoint["gaussian_2d_head"]
    )
    model.to(device).eval()

    dataset = eval(args.dataset)
    loader = get_data_loader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        accelerator=SimpleNamespace(num_processes=1),
        fixed_length=True,
    )
    batch = todevice(next(iter(loader)), device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(batch)

    for view_index in range(min(args.num_views, len(output.ress))):
        pred = output.ress[view_index]
        alpha_mean = None
        if not args.skip_render:
            rendered, depth, alpha, normal, target = render_prediction_view(
                output.views[view_index],
                pred,
            )
            save_image(output_dir / f"view-{view_index:02d}-target.png", target)
            save_image(output_dir / f"view-{view_index:02d}-render.png", rendered)
            save_image(
                output_dir / f"view-{view_index:02d}-comparison.png",
                torch.cat([target, rendered], dim=1),
            )
            save_image(
                output_dir / f"view-{view_index:02d}-alpha.png",
                alpha.unsqueeze(-1).expand(-1, -1, 3),
            )
            save_depth(output_dir / f"view-{view_index:02d}-depth.png", depth)
            save_image(
                output_dir / f"view-{view_index:02d}-normal.png",
                normal * 0.5 + 0.5,
            )
            alpha_mean = alpha.mean().item()
        if args.export_ply:
            surfels = sample_surfels(
                pred,
                batch_index=0,
                stride=args.ply_stride,
                scale_multiplier=args.ply_stride,
            )
            save_surfel_ply(
                output_dir / f"view-{view_index:02d}-gaussians.ply",
                *surfels,
                ply_format=args.ply_format,
                scale_z=args.ply_scale_z,
            )
        print(
            f"view={view_index} "
            f"scale={pred['gaussian_2d_scales'].mean().item():.5f} "
            f"opacity={pred['gaussian_2d_opacity'].mean().item():.4f} "
            + (f"alpha={alpha_mean:.4f}" if alpha_mean is not None else "alpha=skipped")
        )


if __name__ == "__main__":
    main()
