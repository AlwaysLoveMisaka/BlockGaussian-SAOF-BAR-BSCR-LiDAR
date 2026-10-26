#!/usr/bin/env python3

"""
Project aligned LiDAR points into COLMAP camera views.

Outputs for each image:
1. sparse metric depth in camera-space z;
2. valid mask;
3. depth visualization;
4. RGB + LiDAR depth overlay.

The saved supervision remains one-pixel sparse.
Point enlargement is used only in visualization.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("BLOCKGAUSSIAN_ROOT", Path(__file__).resolve().parents[1]))
if not (PROJECT_ROOT / "scene" / "colmap_loader.py").is_file():
    raise FileNotFoundError(
        f"BlockGaussian scene/colmap_loader.py not found under {PROJECT_ROOT}; "
        "set BLOCKGAUSSIAN_ROOT"
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scene.colmap_loader import (  # noqa: E402
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project a local-coordinate LiDAR cache "
            "into COLMAP camera views."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Dataset root containing images/ and sparse/0/."
        ),
    )

    parser.add_argument(
        "--lidar-cache",
        type=Path,
        required=True,
        help=(
            "NPZ LiDAR cache containing xyz with shape [N, 3]."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--image-scale",
        type=float,
        default=0.25,
        help=(
            "Image scale used by BlockGaussian."
        ),
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help=(
            "Number of images to process. "
            "Use 0 to process all registered images."
        ),
    )

    parser.add_argument(
        "--selection",
        choices=[
            "uniform",
            "first",
        ],
        default="uniform",
        help=(
            "How to select images for a partial test."
        ),
    )

    parser.add_argument(
        "--near",
        type=float,
        default=0.1,
        help=(
            "Minimum positive camera-space depth."
        ),
    )

    parser.add_argument(
        "--vis-radius",
        type=int,
        default=2,
        help=(
            "Visualization-only point radius. "
            "Does not modify saved sparse depth or mask."
        ),
    )


    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help=(
            "Skip depth and overlay PNG generation. "
            "Sparse depth NPZ and statistics are still saved."
        ),
    )

    return parser.parse_args()


def select_images(
    image_records,
    num_images,
    mode,
):
    image_records = sorted(
        image_records,
        key=lambda item: item.id,
    )

    total = len(image_records)

    if (
        num_images <= 0
        or num_images >= total
    ):
        return image_records

    if mode == "first":
        return image_records[:num_images]

    indices = np.linspace(
        0,
        total - 1,
        num=num_images,
    )

    indices = np.rint(
        indices
    ).astype(np.int64)

    indices = np.unique(
        indices
    )

    return [
        image_records[index]
        for index in indices
    ]


def get_scaled_intrinsics(
    camera,
    image_scale,
):
    if camera.model != "PINHOLE":
        raise ValueError(
            "Only PINHOLE is currently supported. "
            f"Found camera model: {camera.model}"
        )

    original_width = int(
        camera.width
    )

    original_height = int(
        camera.height
    )

    output_width = int(
        original_width
        * image_scale
    )

    output_height = int(
        original_height
        * image_scale
    )

    scale_x = (
        output_width
        / original_width
    )

    scale_y = (
        output_height
        / original_height
    )

    fx, fy, cx, cy = [
        float(value)
        for value in camera.params
    ]

    fx_scaled = (
        fx
        * scale_x
    )

    fy_scaled = (
        fy
        * scale_y
    )

    cx_scaled = (
        cx
        * scale_x
    )

    cy_scaled = (
        cy
        * scale_y
    )

    return {
        "original_width": (
            original_width
        ),
        "original_height": (
            original_height
        ),
        "output_width": (
            output_width
        ),
        "output_height": (
            output_height
        ),
        "scale_x": (
            scale_x
        ),
        "scale_y": (
            scale_y
        ),
        "fx": (
            fx_scaled
        ),
        "fy": (
            fy_scaled
        ),
        "cx": (
            cx_scaled
        ),
        "cy": (
            cy_scaled
        ),
    }


def project_lidar(
    xyz_world,
    image_record,
    intrinsics,
    near,
):
    rotation_world_to_camera = (
        qvec2rotmat(
            image_record.qvec
        ).astype(
            np.float64
        )
    )

    translation = np.asarray(
        image_record.tvec,
        dtype=np.float64,
    )

    xyz_camera = (
        xyz_world
        @ rotation_world_to_camera.T
        + translation[None, :]
    )

    depth_z = (
        xyz_camera[:, 2]
    )

    front_mask = (
        depth_z > near
    )

    xyz_camera = (
        xyz_camera[
            front_mask
        ]
    )

    depth_z = (
        depth_z[
            front_mask
        ]
    )

    projected_u = (
        intrinsics["fx"]
        * xyz_camera[:, 0]
        / depth_z
        + intrinsics["cx"]
    )

    projected_v = (
        intrinsics["fy"]
        * xyz_camera[:, 1]
        / depth_z
        + intrinsics["cy"]
    )

    pixel_u = np.rint(
        projected_u
    ).astype(
        np.int64
    )

    pixel_v = np.rint(
        projected_v
    ).astype(
        np.int64
    )

    output_width = (
        intrinsics[
            "output_width"
        ]
    )

    output_height = (
        intrinsics[
            "output_height"
        ]
    )

    inside_mask = (
        (pixel_u >= 0)
        & (
            pixel_u
            < output_width
        )
        & (
            pixel_v >= 0
        )
        & (
            pixel_v
            < output_height
        )
    )

    pixel_u = (
        pixel_u[
            inside_mask
        ]
    )

    pixel_v = (
        pixel_v[
            inside_mask
        ]
    )

    depth_z = (
        depth_z[
            inside_mask
        ].astype(
            np.float32
        )
    )

    flat_pixel_index = (
        pixel_v
        * output_width
        + pixel_u
    )

    z_buffer = np.full(
        output_height
        * output_width,
        np.inf,
        dtype=np.float32,
    )

    np.minimum.at(
        z_buffer,
        flat_pixel_index,
        depth_z,
    )

    depth = z_buffer.reshape(
        output_height,
        output_width,
    )

    valid_mask = np.isfinite(
        depth
    )

    depth_output = np.zeros_like(
        depth,
        dtype=np.float32,
    )

    depth_output[
        valid_mask
    ] = depth[
        valid_mask
    ]

    return {
        "depth": (
            depth_output
        ),
        "valid_mask": (
            valid_mask
        ),
        "front_point_count": (
            int(
                np.count_nonzero(
                    front_mask
                )
            )
        ),
        "inside_point_count": (
            int(
                np.count_nonzero(
                    inside_mask
                )
            )
        ),
    }


def create_visualizations(
    image_path,
    depth,
    valid_mask,
    output_size,
    radius,
):
    output_width, output_height = (
        output_size
    )

    with Image.open(
        image_path
    ) as image:
        image_rgb = (
            image.convert(
                "RGB"
            )
        )

        image_rgb = (
            image_rgb.resize(
                (
                    output_width,
                    output_height,
                ),
                resample=(
                    Image.Resampling.LANCZOS
                ),
            )
        )

        image_rgb = np.asarray(
            image_rgb
        )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    normalized = np.zeros(
        depth.shape,
        dtype=np.uint8,
    )

    valid_depth = (
        depth[
            valid_mask
        ]
    )

    if valid_depth.size > 0:

        low = float(
            np.percentile(
                valid_depth,
                2.0,
            )
        )

        high = float(
            np.percentile(
                valid_depth,
                98.0,
            )
        )

        if (
            not np.isfinite(low)
            or not np.isfinite(high)
            or high <= low
        ):
            low = float(
                np.min(
                    valid_depth
                )
            )

            high = float(
                np.max(
                    valid_depth
                )
            )

        if high <= low:
            high = (
                low
                + 1.0
            )

        scaled = (
            (
                depth[
                    valid_mask
                ]
                - low
            )
            / (
                high
                - low
            )
        )

        scaled = np.clip(
            scaled,
            0.0,
            1.0,
        )

        # Near points receive larger values
        # and therefore warmer colors.
        normalized[
            valid_mask
        ] = (
            (
                1.0
                - scaled
            )
            * 255.0
        ).astype(
            np.uint8
        )

    radius = max(
        0,
        int(radius),
    )

    if radius > 0:

        kernel_size = (
            radius
            * 2
            + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        display_values = cv2.dilate(
            normalized,
            kernel,
        )

        display_mask = cv2.dilate(
            (
                valid_mask.astype(
                    np.uint8
                )
                * 255
            ),
            kernel,
        ) > 0

    else:

        display_values = (
            normalized
        )

        display_mask = (
            valid_mask
        )

    color_map = cv2.applyColorMap(
        display_values,
        cv2.COLORMAP_TURBO,
    )

    depth_visualization = (
        np.zeros_like(
            color_map
        )
    )

    depth_visualization[
        display_mask
    ] = color_map[
        display_mask
    ]

    overlay = (
        image_bgr.copy()
    )

    if np.any(
        display_mask
    ):

        blended = (
            0.35
            * overlay[
                display_mask
            ].astype(
                np.float32
            )
            + 0.65
            * color_map[
                display_mask
            ].astype(
                np.float32
            )
        )

        overlay[
            display_mask
        ] = np.clip(
            blended,
            0,
            255,
        ).astype(
            np.uint8
        )

    return (
        depth_visualization,
        overlay,
    )


def make_statistics(
    image_record,
    projection_result,
    intrinsics,
):
    depth = (
        projection_result[
            "depth"
        ]
    )

    valid_mask = (
        projection_result[
            "valid_mask"
        ]
    )

    valid_depth = (
        depth[
            valid_mask
        ]
    )

    valid_pixel_count = int(
        valid_depth.size
    )

    total_pixel_count = int(
        depth.size
    )

    coverage_percent = (
        100.0
        * valid_pixel_count
        / total_pixel_count
    )

    if valid_depth.size > 0:

        depth_min = float(
            np.min(
                valid_depth
            )
        )

        depth_p05 = float(
            np.percentile(
                valid_depth,
                5.0,
            )
        )

        depth_median = float(
            np.median(
                valid_depth
            )
        )

        depth_p95 = float(
            np.percentile(
                valid_depth,
                95.0,
            )
        )

        depth_max = float(
            np.max(
                valid_depth
            )
        )

    else:

        depth_min = float(
            "nan"
        )

        depth_p05 = float(
            "nan"
        )

        depth_median = float(
            "nan"
        )

        depth_p95 = float(
            "nan"
        )

        depth_max = float(
            "nan"
        )

    return {
        "image_id": (
            int(
                image_record.id
            )
        ),
        "image_name": (
            image_record.name
        ),
        "camera_id": (
            int(
                image_record.camera_id
            )
        ),
        "width": (
            int(
                intrinsics[
                    "output_width"
                ]
            )
        ),
        "height": (
            int(
                intrinsics[
                    "output_height"
                ]
            )
        ),
        "front_points": (
            projection_result[
                "front_point_count"
            ]
        ),
        "inside_points": (
            projection_result[
                "inside_point_count"
            ]
        ),
        "valid_pixels": (
            valid_pixel_count
        ),
        "coverage_percent": (
            coverage_percent
        ),
        "depth_min": (
            depth_min
        ),
        "depth_p05": (
            depth_p05
        ),
        "depth_median": (
            depth_median
        ),
        "depth_p95": (
            depth_p95
        ),
        "depth_max": (
            depth_max
        ),
    }


def main():
    args = parse_args()

    dataset_root = (
        args.dataset_root.resolve()
    )

    lidar_cache = (
        args.lidar_cache.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    sparse_dir = (
        dataset_root
        / "sparse"
        / "0"
    )

    image_dir = (
        dataset_root
        / "images"
    )

    npz_dir = (
        output_dir
        / "npz"
    )

    visualization_dir = (
        output_dir
        / "visualization"
    )

    npz_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading LiDAR cache:"
    )

    print(
        lidar_cache
    )

    lidar_data = np.load(
        lidar_cache,
        allow_pickle=False,
    )

    if "xyz" not in lidar_data.files:
        raise KeyError(
            "The LiDAR NPZ does not contain key 'xyz'."
        )

    xyz_world = np.asarray(
        lidar_data["xyz"],
        dtype=np.float64,
    )

    if (
        xyz_world.ndim != 2
        or xyz_world.shape[1] != 3
    ):
        raise ValueError(
            "xyz must have shape [N, 3]. "
            f"Found: {xyz_world.shape}"
        )

    print(
        "LiDAR points:",
        f"{xyz_world.shape[0]:,}",
    )

    cameras = read_intrinsics_binary(
        str(
            sparse_dir
            / "cameras.bin"
        )
    )

    images = read_extrinsics_binary(
        str(
            sparse_dir
            / "images.bin"
        )
    )

    selected_images = select_images(
        list(
            images.values()
        ),
        args.num_images,
        args.selection,
    )

    print(
        "Registered images:",
        len(images),
    )

    print(
        "Selected images:",
        len(
            selected_images
        ),
    )

    print()

    all_statistics = []

    for sequence_index, image_record in enumerate(
        selected_images,
        start=1,
    ):

        camera = cameras[
            image_record.camera_id
        ]

        intrinsics = (
            get_scaled_intrinsics(
                camera,
                args.image_scale,
            )
        )

        print(
            f"[{sequence_index:03d}/"
            f"{len(selected_images):03d}] "
            f"{image_record.name}"
        )

        projection_result = (
            project_lidar(
                xyz_world=(
                    xyz_world
                ),
                image_record=(
                    image_record
                ),
                intrinsics=(
                    intrinsics
                ),
                near=(
                    args.near
                ),
            )
        )

        statistics = (
            make_statistics(
                image_record=(
                    image_record
                ),
                projection_result=(
                    projection_result
                ),
                intrinsics=(
                    intrinsics
                ),
            )
        )

        all_statistics.append(
            statistics
        )

        stem = Path(
            image_record.name
        ).stem

        output_npz = (
            npz_dir
            / f"{stem}.npz"
        )

        np.savez_compressed(
            output_npz,
            depth=(
                projection_result[
                    "depth"
                ].astype(
                    np.float32
                )
            ),
            valid_mask=(
                projection_result[
                    "valid_mask"
                ].astype(
                    np.uint8
                )
            ),
            image_id=(
                np.int32(
                    image_record.id
                )
            ),
            camera_id=(
                np.int32(
                    image_record.camera_id
                )
            ),
            image_name=(
                np.asarray(
                    image_record.name
                )
            ),
            fx=(
                np.float32(
                    intrinsics["fx"]
                )
            ),
            fy=(
                np.float32(
                    intrinsics["fy"]
                )
            ),
            cx=(
                np.float32(
                    intrinsics["cx"]
                )
            ),
            cy=(
                np.float32(
                    intrinsics["cy"]
                )
            ),
            width=(
                np.int32(
                    intrinsics[
                        "output_width"
                    ]
                )
            ),
            height=(
                np.int32(
                    intrinsics[
                        "output_height"
                    ]
                )
            ),
        )

        if args.skip_visualization:

            print(
                "  valid pixels:",
                f"{statistics['valid_pixels']:,}",
            )

            print(
                "  coverage:",
                (
                    f"{statistics['coverage_percent']:.4f}%"
                ),
            )

            print(
                "  visualization: skipped"
            )

            print()

            continue

        image_path = (
            image_dir
            / image_record.name
        )

        (
            depth_visualization,
            overlay,
        ) = create_visualizations(
            image_path=(
                image_path
            ),
            depth=(
                projection_result[
                    "depth"
                ]
            ),
            valid_mask=(
                projection_result[
                    "valid_mask"
                ]
            ),
            output_size=(
                (
                    intrinsics[
                        "output_width"
                    ],
                    intrinsics[
                        "output_height"
                    ],
                )
            ),
            radius=(
                args.vis_radius
            ),
        )

        cv2.imwrite(
            str(
                visualization_dir
                / (
                    f"{stem}"
                    "_lidar_depth.png"
                )
            ),
            depth_visualization,
        )

        cv2.imwrite(
            str(
                visualization_dir
                / (
                    f"{stem}"
                    "_lidar_overlay.png"
                )
            ),
            overlay,
        )

        print(
            "  output size:",
            (
                intrinsics[
                    "output_width"
                ],
                intrinsics[
                    "output_height"
                ],
            ),
        )

        print(
            "  valid pixels:",
            f"{statistics['valid_pixels']:,}",
        )

        print(
            "  coverage:",
            (
                f"{statistics['coverage_percent']:.4f}%"
            ),
        )

        print(
            "  depth median:",
            (
                f"{statistics['depth_median']:.4f}"
            ),
        )

        print()

    csv_path = (
        output_dir
        / "projection_summary.csv"
    )

    fieldnames = list(
        all_statistics[0].keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:

        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_statistics
        )

    json_path = (
        output_dir
        / "projection_summary.json"
    )

    summary = {
        "dataset_root": (
            str(
                dataset_root
            )
        ),
        "lidar_cache": (
            str(
                lidar_cache
            )
        ),
        "image_scale": (
            float(
                args.image_scale
            )
        ),
        "selected_image_count": (
            len(
                selected_images
            )
        ),
        "statistics": (
            all_statistics
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:

        json.dump(
            summary,
            file_handle,
            indent=2,
            ensure_ascii=False,
        )

    coverage_values = np.asarray(
        [
            item[
                "coverage_percent"
            ]
            for item in all_statistics
        ],
        dtype=np.float64,
    )

    print(
        "=" * 72
    )

    print(
        "Projection completed."
    )

    print(
        "Output directory:"
    )

    print(
        output_dir
    )

    print(
        "Coverage mean:",
        (
            f"{coverage_values.mean():.4f}%"
        ),
    )

    print(
        "Coverage min:",
        (
            f"{coverage_values.min():.4f}%"
        ),
    )

    print(
        "Coverage max:",
        (
            f"{coverage_values.max():.4f}%"
        ),
    )

    print(
        "CSV:"
    )

    print(
        csv_path
    )


if __name__ == "__main__":
    main()
