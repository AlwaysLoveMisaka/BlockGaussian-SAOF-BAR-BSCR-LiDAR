import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lidar.sparse_depth_prior import (
    linear_lidar_weight,
    load_inverse_depth_target,
    masked_inverse_depth_l1,
)


class SparseDepthPriorTests(unittest.TestCase):
    def test_linear_weight_schedule(self):
        self.assertEqual(linear_lidar_weight(49, 50, 150, 0.01, 0.03), 0.0)
        self.assertAlmostEqual(linear_lidar_weight(100, 50, 150, 0.01, 0.03), 0.02)
        self.assertAlmostEqual(linear_lidar_weight(200, 50, 150, 0.01, 0.03), 0.03)

    def test_sparse_loading_and_masked_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "view.npz"
            np.savez_compressed(
                path,
                depth=np.array([[2, 4], [0, np.nan]], dtype=np.float32),
                valid_mask=np.ones((2, 2), dtype=np.uint8),
            )
            target, mask = load_inverse_depth_target(path, 2, 2, scene_scale=2)
        self.assertEqual(int(mask.sum()), 2)
        self.assertAlmostEqual(float(target[0, 0]), 0.25)
        self.assertAlmostEqual(float(target[0, 1]), 0.125)
        rendered = torch.tensor([[0.30, 0.10], [5.0, 5.0]])
        loss, count = masked_inverse_depth_l1(rendered, target, mask)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(loss), 0.0375, places=6)
        skipped, count = masked_inverse_depth_l1(rendered, target, mask, min_valid_pixels=3)
        self.assertIsNone(skipped)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
