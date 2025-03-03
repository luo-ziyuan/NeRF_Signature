# NeRF_Signature

Source code of the paper "[The NeRF Signature: Codebook-Aided Watermarking for Neural Radiance Fields](https://arxiv.org/abs/2502.19125)".

## Installation
The project has the same dependencies as [torch-ngp](https://github.com/ashawkey/torch-ngp/tree/main).

## Usage

### 1. Train Clean Models
First, use [torch-ngp](https://github.com/ashawkey/torch-ngp/tree/main) to train clean NeRF models for different datasets:

#### Mip-NeRF 360
```bash
python main_nerf.py data/360_v2/counter --workspace logs/counter_clean -O --scale 0.33 --dt_gamma 0 --tcnn
```

#### Blender
```bash
python main_nerf.py data/nerf_synthetic/hotdog --workspace logs/hotdog_clean -O --bound 1.0 --scale 0.8 --dt_gamma 0 --tcnn
```

#### LLFF
```bash
python main_nerf.py data/nerf_llff_data/fern --workspace logs/fern_clean -O --tcnn
```

#### Tanks and Temple
```bash
python main_nerf.py data/TanksAndTemple/Family --workspace logs/Family_clean -O --bound 1.0 --scale 0.33 --dt_gamma 0 --tcnn
```

After training, move the `.pth` model files to the `./clean_model` directory.

### 2. Train Watermarked Models
Train the signature representation:

#### Mip-NeRF 360
```bash
python main_nerf_wtmk.py data/360_v2/counter --workspace logs/git_counter_wtmk_32b -O --wtmk_tcnn --ckpt ./clean_model/counter_ngp_ep0125.pth --message_dim 32 --loss_w bce --lambda_w 0.005 --lambda_i 1.0 --num_rays 4096 --rand_pose 0 --n_views 1 --iters 1000 --num_rows 32 --num_cols 32 --use_existset --eval_interval 5 --save_interval 5 --num_images_test 10 --scale 0.33 --dt_gamma 0
```

#### Blender
```bash
python main_nerf_wtmk.py data/nerf_synthetic/hotdog --workspace logs/hotdog_wtmk_32b -O --wtmk_tcnn --ckpt ./clean_model/hotdog_ngp_ep0300.pth --message_dim 32 --downscale 2 --loss_w bce --lambda_w 0.005 --lambda_i 1.0 --num_rays 4096 --rand_pose 0 --n_views 1 --iters 1000 --num_rows 32 --num_cols 32 --use_existset --eval_interval 5 --save_interval 5 --num_images_test 10 --bound 1.0 --scale 0.8 --dt_gamma 0
```

#### LLFF
```bash
python main_nerf_wtmk.py data/nerf_llff_data/fern --workspace logs/fern_wtmk_32b -O --wtmk_tcnn --ckpt ./clean_model/fern_ngp_ep1500.pth --message_dim 32 --loss_w bce --lambda_w 0.005 --lambda_i 1.0 --num_rays 4096 --rand_pose 0 --n_views 1 --iters 600 --num_rows 32 --num_cols 32 --use_existset --eval_interval 10 --save_interval 10 --num_images_test 10
```

#### Tanks and Temple
```bash
python main_nerf_wtmk.py data/TanksAndTemple/Family --workspace logs/Family_wtmk_32b -O --wtmk_tcnn --ckpt ./clean_model/Family_ngp_ep0226.pth --message_dim 32 --loss_w bce --lambda_w 0.005 --lambda_i 1.0 --num_rays 4096 --rand_pose 0 --n_views 1 --iters 600 --num_rows 32 --num_cols 32 --use_existset --eval_interval 5 --save_interval 5 --num_images_test 10 --bound 1.0 --scale 0.33 --dt_gamma 0 --downscale 4
```

This project is built upon [torch-ngp](https://github.com/ashawkey/torch-ngp/tree/main) and [HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch). We express our sincere gratitude to the authors of these repositories.

If you find our paper useful for your work please cite:
```
@article{luo2025signature,
  author    = {Ziyuan Luo and Anderson Rocha and Boxin Shi and Qing Guo and Haoliang Li and Renjie Wan},
  title     = {The NeRF Signature: Codebook-Aided Watermarking for Neural Radiance Fields},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2025},
}
```

