# Integrating the sparse LiDAR prior

The reference training patch used the optional projected-LiDAR depth only on training views. This repository publishes the reusable loss/loader code, not a modified copy of the complete upstream `train.py` or `scene_loader.py`.

1. Register the LiDAR point cloud to the COLMAP reconstruction coordinate system and audit scale and occlusion. Run `python lidar/project_lidar_to_sparse_depth.py --help` for projection inputs. Generate targets only for views permitted by the training split.
2. In `SceneDataset._load_single_sample`, look up `<image_stem>.npz` under the chosen LiDAR-depth directory and call `load_inverse_depth_target(npz_path, image_height, image_width, scene_scale, min_valid_pixels)`. Store the returned tensors as `lidar_depth_inv` and `lidar_valid_mask` in the sample. The target array is zero at invalid pixels.
3. In the training loop, after the renderer provides its inverse-depth tensor (the project's `render_pkg["depth"]`), compute the masked L1 term and add the weighted loss:

```python
from lidar.sparse_depth_prior import linear_lidar_weight, masked_inverse_depth_l1

lidar_loss, count = masked_inverse_depth_l1(
    render_pkg["depth"].squeeze(0),
    sample["lidar_depth_inv"].to(device),
    sample["lidar_valid_mask"].to(device),
    min_valid_pixels=minimum_supervised_pixels,
)
if lidar_loss is not None:
    weight = linear_lidar_weight(iteration, start, ramp_end, initial_weight, final_weight)
    loss = loss + weight * lidar_loss
```

The implementation uses a sparse mask; it does not inpaint depth or derive LiDAR normals. The loader uses nearest-neighbor resizing to match the original training patch. Use matching camera intrinsics/resolution and verify the renderer's depth convention before applying the loss. An unverified LiDAR source can reduce reconstruction quality even if training runs normally.
