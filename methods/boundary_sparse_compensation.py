# -*- coding: utf-8 -*-
"""
saof_v2c_boundary_sparse_densify.py

功能说明
--------
本脚本对应项目中的第三个改进步骤：
    边界稀疏补偿精修（Boundary Sparse Compensation and Refinement, BSCR）

它的作用不是重新训练完整模型，而是在已经完成“边界外观精修”的模型基础上，
针对 block 内部边界附近表达能力不足的稀疏区域，少量复制（clone）已有 Gaussian，
并对这些新增 Gaussian 做轻微扰动、降低初始透明度、缩小初始尺度。

整体流程可以概括为：

    1. 读取上一阶段输出的 point_cloud_merged.ply；
    2. 根据 blocks_info.json 找到内部 block 边界；
    3. 找出靠近内部边界的 Gaussian；
    4. 用 voxel density 估计边界区域的局部稀疏程度；
    5. 优先选择“更稀疏、且更靠近内部边界”的 Gaussian；
    6. 复制少量 Gaussian 作为边界补偿点；
    7. 对新增点做轻微 jitter，避免完全重叠；
    8. 降低新增点 opacity，缩小新增点 scale；
    9. 保存新的 point_cloud_merged.ply；
   10. 后续再调用 saof_v2a_boundary_refine.py 对边界区域做局部精修。

注意
----
本脚本只负责“补点”，不负责后续优化。
完整 v2c / BSCR 流程一般是：

    saof_v2c_boundary_sparse_densify.py
        → 生成补点后的 point_cloud_merged.ply

    saof_v2a_boundary_refine.py
        → 在补点后的模型上继续做 boundary-only refinement

设计动机
--------
v2a / 边界外观精修只能调整已有 Gaussian 的颜色和透明度。
如果某些边界区域本身 Gaussian 表达支撑不足，仅靠调整已有点可能不够。
因此 v2c 进一步在边界稀疏区域补充极少量 Gaussian，使边界区域获得更好的表达能力。

与普通 densification 的区别
--------------------------
本脚本不是全局加点，也不是随机加点，而是：
    - 只在 internal block boundary 附近找点；
    - 只选择 voxel density 较低的稀疏区域；
    - 新增点数量由 add_ratio 控制，通常非常小；
    - 新增点初始 opacity 较低，scale 略小，避免破坏已有结果；
    - 后续只做边界局部 refinement，不扰动全局模型。

典型调用示例
------------
python -u tools/saof_v2c_boundary_sparse_densify.py \
  --src-out output/saof_v2a_step_sweep_w6_lr02_round2/w6_lr02_s1600 \
  --dst-out output/KEEP_saof_v2c_dense_w6_vox2_add000075 \
  --blocks-info output/rubble_b2_60k_3090_labaxis_roi4_bs2/blocks_info.json \
  --boundary-width 6.0 \
  --add-ratio 0.000075 \
  --voxel-size 2.0 \
  --opacity-logit-delta -2.0 \
  --scale-log-delta -0.2 \
  --jitter-scale 0.5 \
  --jitter-max 0.5 \
  --seed 0
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def get_internal_planes(blocks_info_path):
    """
    从 blocks_info.json 中提取 internal block boundary 的世界坐标位置。

    BlockGaussian 会把大场景划分成多个 block。
    每个 block 都有一个 bounding box，即 bbx。
    相邻 block 的边界在世界坐标中会共享某些 x 或 z 平面。

    举例：
        block A 的 xmax 和 block B 的 xmin 相同，
        那么这个 x 值就是一个内部边界平面。

    这里的处理方式：
        1. 读取所有 block 的 bbx；
        2. 收集每个 block 在 x 方向的 min / max；
        3. 收集每个 block 在 z 方向的 min / max；
        4. 去重排序；
        5. 去掉最外侧边界，只保留中间边界；
        6. 返回 internal_x 和 internal_z。

    为什么去掉最外侧？
        最外侧是整个场景的外边界，不是 block 与 block 之间的内部接缝。
        本方法只希望修复 block 内部交界处，不希望在场景外边界补点。

    参数
    ----
    blocks_info_path : str 或 Path
        blocks_info.json 的路径。

    返回
    ----
    internal_x : list[float]
        x 方向的内部边界平面坐标。
    internal_z : list[float]
        z 方向的内部边界平面坐标。
    """
    with open(blocks_info_path, "r") as f:
        info = json.load(f)

    xs, zs = [], []

    # 遍历所有 block，提取每个 block 的 x/z 边界范围。
    for k, v in info.items():
        if k == "num_blocks":
            continue

        bbx = np.asarray(v["bbx"], dtype=np.float64)

        # bbx 是 8 个角点。取 x 和 z 的最小/最大值即可得到该 block 的水平范围。
        xs += [float(bbx[:, 0].min()), float(bbx[:, 0].max())]
        zs += [float(bbx[:, 2].min()), float(bbx[:, 2].max())]

    def unique_sorted(vals):
        """
        对边界坐标排序并去重。

        使用 1e-4 容差是为了避免浮点数误差导致相同边界被当成不同边界。
        """
        vals = sorted(vals)
        out = []
        for x in vals:
            if not out or abs(x - out[-1]) > 1e-4:
                out.append(x)
        return out

    ux = unique_sorted(xs)
    uz = unique_sorted(zs)

    # 去掉首尾，即去掉整个场景的最外侧边界，只保留内部 block 分界线。
    internal_x = ux[1:-1] if len(ux) > 2 else []
    internal_z = uz[1:-1] if len(uz) > 2 else []

    return internal_x, internal_z


def make_boundary_mask(xyz, internal_x, internal_z, width):
    """
    找出距离内部边界足够近的 Gaussian。

    对每个 Gaussian，计算它到所有 internal_x / internal_z 平面的最近距离。
    如果最近距离小于 boundary_width，就认为它属于边界区域。

    这里使用的是 x-z 平面距离：
        - x 方向内部边界：abs(x - internal_x)
        - z 方向内部边界：abs(z - internal_z)

    y 方向不参与判断，因为 block 的分割主要发生在水平 x-z 平面上。
    对于 aerial / rubble 这类大场景，x-z 平面边界更符合 block fusion 的主要问题区域。

    参数
    ----
    xyz : ndarray, shape = [N, 3]
        所有 Gaussian 的世界坐标。
    internal_x : list[float]
        x 方向内部边界。
    internal_z : list[float]
        z 方向内部边界。
    width : float
        边界区域宽度。距离内部边界小于该值的点会被选中。

    返回
    ----
    boundary_mask : ndarray[bool], shape = [N]
        True 表示该点属于边界区域。
    dist : ndarray[float], shape = [N]
        每个点到最近内部边界的距离。
    """
    # 初始化为无穷大，后续不断取更小的距离。
    dist = np.full((xyz.shape[0],), np.inf, dtype=np.float32)

    # 到 x 方向内部边界的距离。
    for px in internal_x:
        dist = np.minimum(dist, np.abs(xyz[:, 0] - float(px)))

    # 到 z 方向内部边界的距离。
    for pz in internal_z:
        dist = np.minimum(dist, np.abs(xyz[:, 2] - float(pz)))

    # 小于 width 的点被认为是 boundary Gaussian。
    return dist <= float(width), dist


def structured_voxel_inverse(coords, voxel_size):
    """
    用 3D voxel density 近似估计局部稀疏程度。

    为什么不用真正 kNN？
        对大规模 Gaussian 场景来说，点数通常是千万级。
        如果直接对所有边界点做 kNN，计算和内存开销都会很大。
        因此这里用 voxel density 作为轻量替代。

    核心思想：
        1. 把边界区域的 Gaussian 坐标划分到 3D voxel 网格中；
        2. 统计每个 voxel 里有多少个点；
        3. 每个点的 density 就是它所在 voxel 的点数；
        4. density 越小，说明该点周围越稀疏；
        5. 优先在这些稀疏区域 clone Gaussian。

    参数
    ----
    coords : ndarray, shape = [N, 3]
        边界区域 Gaussian 的坐标。
    voxel_size : float
        voxel 的边长。越小越敏感，越大越平滑。

    返回
    ----
    density : ndarray[int], shape = [N]
        每个点所在 voxel 的点数。
        数值越小表示越稀疏。
    """
    # 以边界点自身的最小坐标为 voxel 网格原点。
    mins = coords.min(axis=0, keepdims=True)

    # 将连续坐标映射到整数 voxel index。
    vox = np.floor((coords - mins) / float(voxel_size)).astype(np.int64)
    vox = np.ascontiguousarray(vox)

    # 把每个 voxel 的三维 index 变成结构化 dtype，
    # 这样 np.unique 可以把 (x, y, z) 三元组当成一个整体进行去重统计。
    dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    vox_view = vox.view(dtype).reshape(-1)

    # inv: 每个点属于哪个唯一 voxel；
    # counts: 每个唯一 voxel 中包含多少个点。
    _, inv, counts = np.unique(vox_view, return_inverse=True, return_counts=True)

    # 每个点的 density = 它所在 voxel 的点数。
    density = counts[inv]
    return density


def main():
    """
    主函数：执行边界稀疏区域补点。

    输入：
        --src-out:
            上一阶段模型目录。通常是 v2a / 边界外观精修的输出目录。
            该目录下必须有 point_cloud_merged.ply 和 config.yaml。

        --dst-out:
            本脚本输出目录。
            会生成补点后的 point_cloud_merged.ply 和 SAOF_V2C_DENSIFY_SUMMARY.json。

        --blocks-info:
            原始 baseline 的 blocks_info.json，用于确定内部 block 边界。

    输出：
        dst_out/point_cloud_merged.ply
            补点后的完整 Gaussian 模型。

        dst_out/SAOF_V2C_DENSIFY_SUMMARY.json
            补点参数和统计信息。
    """
    ap = argparse.ArgumentParser()

    # 输入输出目录。
    ap.add_argument("--src-out", required=True)
    ap.add_argument("--dst-out", required=True)

    # blocks_info.json 用于定位 internal boundary。
    ap.add_argument("--blocks-info", required=True)

    # 边界区域宽度。距离 internal boundary 小于该值的 Gaussian 会被纳入候选范围。
    ap.add_argument("--boundary-width", type=float, default=6.0)

    # 新增点比例。实际新增点数 add_n = round(total_points * add_ratio)。
    # 注意这是相对全模型点数的比例，不是相对 boundary 点数。
    ap.add_argument("--add-ratio", type=float, default=0.0002)

    # voxel density 的网格尺寸，用于判断局部稀疏程度。
    ap.add_argument("--voxel-size", type=float, default=2.0)

    # 新增点的 opacity logit 偏移。
    # 负数表示降低新增点初始透明度，避免一开始对渲染结果产生过强影响。
    ap.add_argument("--opacity-logit-delta", type=float, default=-2.0)

    # 新增点的 scale log 偏移。
    # 负数表示缩小新增点尺度，避免边界补点造成模糊或扩散。
    ap.add_argument("--scale-log-delta", type=float, default=-0.2)

    # 新增点坐标扰动比例。
    # 扰动会按原 Gaussian 的 scale 缩放，使 clone 点不完全重合。
    ap.add_argument("--jitter-scale", type=float, default=0.2)

    # 单个坐标方向的扰动最大值，防止新增点偏离太远。
    ap.add_argument("--jitter-max", type=float, default=0.05)

    # 随机种子，保证补点选择后的 jitter 可复现。
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # 使用 numpy 的随机数生成器，保证随机扰动可控。
    rng = np.random.default_rng(args.seed)

    src_out = Path(args.src_out)
    dst_out = Path(args.dst_out)
    if src_out.resolve() == dst_out.resolve():
        ap.error("--src-out and --dst-out must differ")

    src_ply = src_out / "point_cloud_merged.ply"
    src_cfg = src_out / "config.yaml"

    # 补点必须从已经融合后的完整模型开始。
    if not src_ply.exists():
        raise FileNotFoundError(src_ply)
    if not src_cfg.exists():
        raise FileNotFoundError(src_cfg)

    # Public release safety: never destroy an existing model or candidate.
    if dst_out.exists():
        raise FileExistsError(f"Refusing to overwrite existing BSCR output: {dst_out}")
    dst_out.mkdir(parents=True, exist_ok=False)

    # 复制 config.yaml，保证后续 render_views.py / metrics.py 可以直接使用该目录。
    shutil.copy2(src_cfg, dst_out / "config.yaml")

    # 复制上游阶段的记录文件，方便追踪实验来源。
    for name in [
        "blocks_info.json",
        "SAOF_SUMMARY.json",
        "SAOF_V2A_SUMMARY.json",
        "SAOF_V2A_FINAL_RESULT_SUMMARY.txt",
    ]:
        p = src_out / name
        if p.exists():
            shutil.copy2(p, dst_out / name)

    print("[SAOF-v2c] loading:", src_ply)

    # 读取 PLY 文件。
    # vertices 是结构化数组，字段包括：
    # x, y, z, nx, ny, nz, f_dc_*, f_rest_*, opacity, scale_*, rot_* 等。
    ply = PlyData.read(str(src_ply))
    vertices = ply["vertex"].data

    # 提取 Gaussian 中心坐标。
    x = vertices["x"].astype(np.float32)
    y = vertices["y"].astype(np.float32)
    z = vertices["z"].astype(np.float32)
    xyz = np.stack([x, y, z], axis=1)

    # 1. 找内部 block 边界；
    # 2. 找靠近内部边界的 Gaussian。
    internal_x, internal_z = get_internal_planes(args.blocks_info)
    boundary_mask, boundary_dist = make_boundary_mask(
        xyz, internal_x, internal_z, args.boundary_width
    )
    boundary_idx = np.where(boundary_mask)[0]

    n_total = len(vertices)
    n_boundary = len(boundary_idx)

    # 计算要新增的点数。
    # add_ratio 是相对总点数的比例；
    # 但新增点最多不能超过 boundary 点数量。
    add_n = max(1, int(round(n_total * args.add_ratio)))
    add_n = min(add_n, n_boundary)

    print("[SAOF-v2c] internal_x:", internal_x)
    print("[SAOF-v2c] internal_z:", internal_z)
    print("[SAOF-v2c] total points:", n_total)
    print("[SAOF-v2c] boundary points:", n_boundary)
    print("[SAOF-v2c] boundary ratio:", n_boundary / max(n_total, 1))
    print("[SAOF-v2c] add_ratio:", args.add_ratio)
    print("[SAOF-v2c] add_n:", add_n)
    print("[SAOF-v2c] voxel_size:", args.voxel_size)

    # 近似 Denoising-GS 的 kNN 稀疏度：
    # 大场景直接算 kNN 代价过高，这里用 3D voxel density 作为 sparse proxy。
    boundary_xyz = xyz[boundary_idx]
    density = structured_voxel_inverse(boundary_xyz, args.voxel_size)

    # 选择要 clone 的边界 Gaussian。
    #
    # np.lexsort 的排序规则：
    #   最后一个 key 是主排序 key。
    #
    # 这里写作：
    #   np.lexsort((boundary_dist[boundary_idx], density))
    #
    # 表示：
    #   1. 先按 density 从小到大排序；
    #   2. density 相同时，再按 boundary_dist 从小到大排序。
    #
    # 也就是：
    #   density 越小 → 越稀疏 → 越优先；
    #   同样稀疏时，越靠近 internal boundary → 越优先。
    local_order = np.lexsort((boundary_dist[boundary_idx], density))
    chosen = boundary_idx[local_order[:add_n]]

    print(
        "[SAOF-v2c] selected density min/mean/max:",
        int(density[local_order[:add_n]].min()),
        float(density[local_order[:add_n]].mean()),
        int(density[local_order[:add_n]].max()),
    )

    # 复制被选中的 Gaussian。
    # 这里 copy 的是结构化数组，因此会完整复制该 Gaussian 的所有属性：
    #   position, color SH, opacity, scale, rotation 等。
    clones = vertices[chosen].copy()

    # 轻微扰动克隆点，避免新增点与原点完全重叠。
    #
    # 如果完全重叠，新增点和原点可能表达冗余；
    # 轻微 jitter 可以让新增点在局部稀疏区域提供一点新的空间支撑。
    has_scale = all(f"scale_{i}" in vertices.dtype.names for i in range(3))

    if has_scale and args.jitter_scale > 0:
        # 读取原 Gaussian 的 log-scale。
        scales_log = np.stack(
            [vertices[f"scale_{i}"][chosen].astype(np.float32) for i in range(3)],
            axis=1,
        )

        # 3DGS 中 scale 通常以 log 形式存储，这里 exp 还原到真实尺度。
        scales = np.exp(scales_log)

        # 生成标准正态扰动。
        noise = rng.normal(0.0, 1.0, size=(add_n, 3)).astype(np.float32)

        # 扰动幅度与原 Gaussian scale 成比例。
        jitter = noise * scales * float(args.jitter_scale)

        # 限制扰动最大幅度，避免新增点偏移过远。
        jitter = np.clip(jitter, -float(args.jitter_max), float(args.jitter_max))
    else:
        # 如果没有 scale 字段，或者 jitter_scale <= 0，则不做扰动。
        jitter = np.zeros((add_n, 3), dtype=np.float32)

    # 应用坐标扰动。
    clones["x"] = clones["x"] + jitter[:, 0]
    clones["y"] = clones["y"] + jitter[:, 1]
    clones["z"] = clones["z"] + jitter[:, 2]

    # 新增点 opacity 降低。
    #
    # opacity 在 3DGS 的 PLY 中通常存的是 logit，不是 0~1 概率。
    # 所以这里直接加一个负的 logit delta。
    #
    # 这样做的原因：
    #   新增点刚被 clone 出来时，还没有经过后续 refinement 校准；
    #   如果 opacity 太高，可能直接影响渲染，造成重影或过亮；
    #   因此先低透明度初始化，后续再由 v2a refine 调整。
    if "opacity" in vertices.dtype.names:
        clones["opacity"] = clones["opacity"] + float(args.opacity_logit_delta)

    # 新增点 scale 略微缩小。
    #
    # scale 过大容易造成模糊扩散；
    # 对边界补点来说，更希望新增点提供局部细粒度支撑，而不是大面积覆盖。
    if has_scale:
        for i in range(3):
            clones[f"scale_{i}"] = clones[f"scale_{i}"] + float(args.scale_log_delta)

    # 将原始点和新增点拼接成新的 vertex 数组。
    combined = np.empty(n_total + add_n, dtype=vertices.dtype)
    combined[:n_total] = vertices
    combined[n_total:] = clones

    # 重建 PLY elements。
    #
    # 一般 PLY 里最重要的是 vertex element；
    # 其他 element 如果存在，保持原样复制。
    elements = []
    for elem in ply.elements:
        if elem.name == "vertex":
            elements.append(PlyElement.describe(combined, "vertex"))
        else:
            elements.append(elem)

    # 保存补点后的完整 Gaussian 模型。
    out_ply = dst_out / "point_cloud_merged.ply"
    out = PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    out.write(str(out_ply))

    # 保存实验摘要，方便后续确认补点参数和点数变化。
    summary = {
        "method": "SAOF-v2c boundary-aware spatial coherence densification",
        "src_out": str(src_out),
        "dst_out": str(dst_out),
        "blocks_info": str(args.blocks_info),
        "boundary_width": args.boundary_width,
        "add_ratio": args.add_ratio,
        "voxel_size": args.voxel_size,
        "opacity_logit_delta": args.opacity_logit_delta,
        "scale_log_delta": args.scale_log_delta,
        "jitter_scale": args.jitter_scale,
        "jitter_max": args.jitter_max,
        "seed": args.seed,
        "total_points_before": int(n_total),
        "boundary_points": int(n_boundary),
        "boundary_ratio": float(n_boundary / max(n_total, 1)),
        "added_points": int(add_n),
        "total_points_after": int(n_total + add_n),
        "internal_x": internal_x,
        "internal_z": internal_z,
    }

    (dst_out / "SAOF_V2C_DENSIFY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("[SAOF-v2c] saved:", out_ply)
    print("[SAOF-v2c] saved:", dst_out / "SAOF_V2C_DENSIFY_SUMMARY.json")
    print("[SAOF-v2c] done")


if __name__ == "__main__":
    main()
