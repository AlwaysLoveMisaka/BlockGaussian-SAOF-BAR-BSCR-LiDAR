# BlockGaussian: SAOF, BAR, BSCR and sparse LiDAR guidance

Research code for post-training refinement of block-based Gaussian splats and an optional projected-LiDAR sparse-depth prior. This repository contains our method scripts, **not** the upstream BlockGaussian implementation, datasets, LiDAR point clouds, trained models, or experiment logs.

This release is designed to be used with [BlockGaussian](https://github.com/SunshineWYC/BlockGaussian). Set `BLOCKGAUSSIAN_ROOT` to an existing checkout with its environment installed. The post-training scripts expect its `config.yaml`, `blocks_info.json`, and Gaussian PLY layout. See [upstream attribution](docs/UPSTREAM.md).

## Post-training stages

1. **SAOF** (`methods/seam_aware_overlap_fusion.py`): preserve a weighted overlap band and fuse per-block Gaussian models.
2. **BAR** (`methods/boundary_appearance_refinement.py`): refine boundary Gaussian appearance while leaving the rest of the scene fixed.
3. **BSCR** (`methods/boundary_sparse_compensation.py`): add a small number of boundary Gaussians in locally sparse regions.
4. **Post-BSCR BAR**: run the BAR script again on the BSCR output. BSCR alone is not the final pipeline.

`methods/boundary_segments.py` is shared by all three stages. Each stage requires a *new* output directory and refuses to overwrite an existing candidate.

Example sequence (substitute your own paths and tuned values):

```bash
export BLOCKGAUSSIAN_ROOT=/path/to/BlockGaussian
python methods/seam_aware_overlap_fusion.py --src-out /path/to/baseline --dst-out /path/to/saof --iteration 30000 --mode ownership --overlap-width 1.0 --min-weight 0.1 --extra-keep-opacity 0.005
python methods/boundary_appearance_refinement.py --src-out /path/to/saof --dst-out /path/to/bar --blocks-info /path/to/baseline/blocks_info.json --boundary-width 45.0 --steps 1600 --lr-scale 0.4
python methods/boundary_sparse_compensation.py --src-out /path/to/bar --dst-out /path/to/bscr --blocks-info /path/to/baseline/blocks_info.json --boundary-width 45.0 --add-ratio 0.00005 --voxel-size 2.0 --opacity-logit-delta -2.0 --scale-log-delta -0.2 --jitter-scale 0.2 --jitter-max 0.05
python methods/boundary_appearance_refinement.py --src-out /path/to/bscr --dst-out /path/to/bscr_bar --blocks-info /path/to/baseline/blocks_info.json --boundary-width 45.0 --steps 3200 --lr-scale 0.4
```

The numeric choices above reflect one TUM2TWIN experiment, not recommended defaults for other scenes. The original sweep orchestrators were deliberately excluded because they hard-code a server's paths and experiment-specific candidate selection.

## Optional LiDAR depth prior

`lidar/project_lidar_to_sparse_depth.py` projects aligned LiDAR points into COLMAP training cameras and saves sparse camera-Z depth and a validity mask. `lidar/sparse_depth_prior.py` loads this data, converts it to inverse depth in the model's scale, computes a masked L1 loss, and implements the loss-weight ramp used in our training patch. [Integration notes](docs/LIDAR_INTEGRATION.md) show where to call it in BlockGaussian.

**The released LiDAR guidance is a sparse depth prior, not a LiDAR-normal loss.** Validate LiDAR-to-camera registration, visibility, depth scale, and train/test separation before enabling supervision. Merely projecting LiDAR into matching image coordinates does not establish geometrically correct depth targets.

## Status and reproducibility

The post-training method scripts are extracted from our experimental implementation. Static syntax checks pass; a clean end-to-end run of this standalone package against a fresh upstream checkout has **not** yet been verified. It is a research-code release, not a turnkey reproduction of the paper results. The TUM2TWIN experiment used three BlockGaussian spatial blocks; `split8` referred to an every-eighth-view evaluation split, not eight blocks.

No license is granted by this repository at present. Upstream BlockGaussian and its submodules have separate authors and terms; they are not copied here.
