# Extended Gaussian Training Module

This repository extends the frozen TTT3R/CUT3R reconstruction pipeline with DPT-based Gaussian heads.

Current main modules:

- `DPT-2DGS`: predicts 2D Gaussian / surfel parameters with multi-view render supervision.
- `DPT-GS`: predicts 3D Gaussian parameters with the same training entry point style.
- `render`: renders trained Gaussian heads to RGB, alpha, depth, normal, comparison images, and optional PLY files.
- `demo.py`: keeps the original TTT3R online reconstruction demo entry point.

## Installation

```bash
conda create -n ttt3r python=3.11 cmake=3.14.0
conda activate ttt3r

pip install -r requirements.txt
```

Compile the RoPE CUDA extension:

```bash
cd src/croco/models/curope
python setup.py build_ext --inplace
cd ../../../..
```

If the system CUDA version is different from the CUDA version used by PyTorch, make sure `nvcc` comes from the active conda environment.

## Checkpoint

The default backbone checkpoint is the CUT3R DPT checkpoint:

```bash
cd src
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ..
```

Default path:

```text
src/cut3r_512_dpt_4_64.pth
```

## TTT3R Demo

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --size 512 \
  --seq_path examples/taylor.mp4 \
  --output_dir tmp/taylor \
  --port 8080 \
  --model_update_type ttt3r \
  --frame_interval 1 \
  --reset_interval 50 \
  --downsample_factor 100 \
  --vis_threshold 10.0
```

## DPT-2DGS Training

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_2d_gaussian_ddp.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --train_dataset "Replica_Multi(ROOT='/home/slam/test/nice-slam/Datasets/Replica',split='train',resolution=512,num_views=2,aug_crop=0,image_dir='results',max_interval=4,scene_names='office0')" \
  --output_dir output/dpt-2dgs-replica-office0 \
  --epochs 10 \
  --steps_per_epoch 5000 \
  --batch_size 1 \
  --num_workers 2 \
  --lr 0.0002
```

Multi-GPU DDP:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node 4 train_2d_gaussian_ddp.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --train_dataset "Replica_Multi(ROOT='/data3/junzebao/Replica',split='train',resolution=512,num_views=2,aug_crop=0,image_dir='results',max_interval=4)" \
  --output_dir output/dpt-2dgs-replica-ddp \
  --epochs 10 \
  --steps_per_epoch 5000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 0.0002
```

Resume training:

```bash
--resume output/dpt-2dgs-replica-ddp/2d-gaussian-head-last.pth
```

## DPT-2DGS Rendering

```bash
CUDA_VISIBLE_DEVICES=0 python render_2d_gaussian.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --gaussian_checkpoint output/dpt-2dgs-replica-office0/2d-gaussian-head-last.pth \
  --dataset "Replica_Multi(ROOT='/home/slam/test/nice-slam/Datasets/Replica',split='train',resolution=512,num_views=2,aug_crop=0,image_dir='results',max_interval=4,scene_names='office0')" \
  --output_dir output/dpt-2dgs-replica-office0/renders/office0 \
  --export_ply
```

The renderer writes:

```text
view-00-target.png
view-00-render.png
view-00-comparison.png
view-00-alpha.png
view-00-depth.png
view-00-normal.png
view-00-gaussians.ply
```

## DPT-GS Training

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_gaussian_ddp.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --train_dataset "Replica_Multi(ROOT='/home/slam/test/nice-slam/Datasets/Replica',split='train',resolution=512,num_views=2,aug_crop=0,image_dir='results',max_interval=4,scene_names='office0')" \
  --output_dir output/dpt-gs-replica-office0 \
  --epochs 10 \
  --steps_per_epoch 5000 \
  --batch_size 1 \
  --num_workers 2 \
  --lr 0.0002
```

Multi-GPU DDP:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node 4 train_gaussian_ddp.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --train_dataset "Replica_Multi(ROOT='/data3/junzebao/Replica',split='train',resolution=512,num_views=2,aug_crop=0,image_dir='results',max_interval=4)" \
  --output_dir output/dpt-gs-replica-ddp \
  --epochs 10 \
  --steps_per_epoch 5000 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 0.0002
```

## DPT-GS Rendering

```bash
CUDA_VISIBLE_DEVICES=0 python render_gaussian.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --gaussian_checkpoint output/dpt-gs-replica-office0/gaussian-head-last.pth \
  --dataset "Replica_Multi(ROOT='/home/slam/test/nice-slam/Datasets/Replica',split='train',resolution=512,num_views=2,aug_crop=0,image_dir='results',max_interval=4,scene_names='office0')" \
  --output_dir output/dpt-gs-replica-office0/renders/office0 \
  --export_ply
```

## Common Arguments

- `epochs`: train until this epoch index. With `--resume`, training starts from the next epoch stored in the checkpoint.
- `steps_per_epoch`: maximum optimizer steps per epoch. Use `0` to run the full dataloader.
- `batch_size`: batch size per GPU. In DDP, the effective global batch is approximately `batch_size * number_of_gpus * accum_iter`.
- `num_views`: set inside the dataset string; controls how many frames are sampled per training item.
- `max_interval`: set inside the dataset string; controls the maximum temporal gap between sampled views.
- `render_stride`: rendering downsampling stride. The default `1` uses full resolution.

## Acknowledgements

- [CUT3R](https://github.com/CUT3R/CUT3R)
- [TTT3R](https://github.com/Inception3D/TTT3R)

## License

This repository contains upstream CUT3R/TTT3R code and local DPT-GS / DPT-2DGS extensions. See [LICENSE](LICENSE) for upstream license notices, non-commercial use restrictions, and patent reservation terms for the local extensions.
