"""
seam_aware_overlap_fusion.py

用途：
    这是 BlockGaussian 改进流程中的 v1 方法：边界重叠柔性融合
    （Seam-Aware Overlap Fusion）。

核心思想：
    原始 BlockGaussian 在融合阶段通常会把每个 block 裁剪到自己的 bbox 内，
    然后直接把各 block 的 Gaussian 点云拼接起来。这样做很快，但在 block
    内部边界处容易出现“硬切”的感觉。

    本脚本的做法是：
    1. 找出相邻 block 之间的内部边界；
    2. 在内部边界附近保留一小段 overlap band；
    3. 根据 Gaussian 到边界的距离调整 opacity 权重；
    4. 原始 bbox 内的核心点始终保留；
    5. bbox 外但位于 overlap 区域的辅助点，只有足够有效才保留；
    6. 最后把处理后的所有 block 合并成 point_cloud_merged.ply。

输入：
    --src-out：原始 BlockGaussian baseline 输出目录。
    该目录下需要包含：
        config.yaml
        blocks_info.json
        point_cloud/<block_id>/point_cloud_60000.ply

输出：
    --dst-out：SAOF 处理后的输出目录。
    该目录下会生成：
        config.yaml
        blocks_info.json
        point_cloud_postfix/point_cloud_<block_id>.ply
        point_cloud_merged.ply
        SAOF_SUMMARY.json

注意：
    运行本脚本后，不要再对 dst-out 运行原始 block_fusion.py，
    否则 point_cloud_postfix 和 point_cloud_merged.ply 可能被覆盖。
"""

# 标准库：用于命令行参数、JSON/YAML 文件读写、路径管理和目录复制。
import argparse
import json
import os
import re  # 当前版本中未直接使用，保留不影响运行。
import shutil
import sys
from pathlib import Path

# 第三方库：numpy 负责点云数组计算，yaml 负责读取/写入 config.yaml。
import numpy as np
import yaml

# 将 BlockGaussian 项目根目录加入 Python 搜索路径。
# 这样脚本即使放在 tools/ 目录下，也可以正常导入项目内部模块。
PROJECT_ROOT = Path(os.environ.get("BLOCKGAUSSIAN_ROOT", Path(__file__).resolve().parents[1]))
if not PROJECT_ROOT.is_dir():
    raise FileNotFoundError(f"BlockGaussian repository not found: {PROJECT_ROOT}; set BLOCKGAUSSIAN_ROOT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PLY 读写工具：用于保存 Gaussian 点云文件。
from plyfile import PlyData, PlyElement

# BlockGaussian 原项目中的工具函数，用来读取 3D Gaussian 的 ply 属性。
from utils.utils import load_gs_ply
from boundary_segments import extract_segments, mask_numpy


def sigmoid(x):
    """
    将 opacity 的 logit 表示转换为概率表示。

    在 3D Gaussian Splatting 中，opacity 通常以 logit 形式存储。
    例如：
        logit 很大  -> opacity 接近 1
        logit 很小  -> opacity 接近 0

    这里先对输入做 clip，是为了避免 exp 溢出。
    """
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    """
    将 opacity 的概率表示转换回 logit 表示。

    本脚本会先把 opacity 转成概率，乘以 SAOF 的融合权重，
    然后再转换回 logit 保存到 ply 文件中。

    对 p 做 clip 是为了避免 log(0) 或除以 0。
    """
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def calculate_block_transform(block_bbx):
    """
    根据一个 block 的 8 个 bbox 顶点，计算世界坐标到 block 局部坐标的变换。

    参数：
        block_bbx:
            shape = [8, 3]，表示 block 的 8 个包围盒顶点，坐标在世界坐标系中。

    返回：
        W2B:
            World-to-Block 变换矩阵。
            用它可以把世界坐标 xyz 转换到当前 block 的局部坐标系。
        x_extent, y_extent, z_extent:
            当前 block 在局部坐标系下的半宽度/半高度/半深度。

    作用：
        原始 block_fusion.py 也是先把点变换到 block 坐标系，
        然后判断点是否在 bbox 范围内。本脚本保持同样的 y 方向裁剪逻辑，
        同时在 x/z 方向加入内部边界 overlap 处理。
    """
    A, B, C, D = block_bbx[:4, :]
    E, F, G, H = block_bbx[4:, :]
    center = (A + G) / 2

    axis_x = (B - A) / np.linalg.norm((B - A))
    axis_z = (D - A) / np.linalg.norm((D - A))
    axis_y = (E - A) / np.linalg.norm((E - A))

    B2W = np.zeros((4, 4), dtype=np.float32)
    B2W[:3, 0], B2W[:3, 1], B2W[:3, 2] = axis_x, axis_y, axis_z
    B2W[:3, 3] = center
    B2W[3, 3] = 1.0

    W2B = np.linalg.inv(B2W)
    vertices_block = (W2B[:3, :3] @ block_bbx.T).T + W2B[:3, 3]
    x_extent = abs(vertices_block[0, 0])
    y_extent = abs(vertices_block[0, 1])
    z_extent = abs(vertices_block[0, 2])

    return W2B.astype(np.float32), float(x_extent), float(y_extent), float(z_extent)


def save_gs_ply(plypath, xyz, features_dc, features_rest, opacity, scaling, rotation):
    """
    保存 3D Gaussian 点云到 ply 文件。

    每个 Gaussian 需要保存的主要属性包括：
        xyz:
            Gaussian 中心位置。
        features_dc:
            SH 的 DC 颜色分量，可以粗略理解为基础颜色。
        features_rest:
            SH 的高阶颜色分量，用于视角相关外观。
        opacity:
            不透明度，注意这里保存的是 logit 格式。
        scaling:
            Gaussian 的尺度。
        rotation:
            Gaussian 的旋转。

    注意：
        ply 字段名必须与 BlockGaussian / 3DGS 读取逻辑一致，
        否则后续 render_views.py 或 GaussianModel.load_ply 可能无法读取。
    """
    os.makedirs(os.path.dirname(plypath), exist_ok=True)

    normals = np.zeros_like(xyz)
    f_dc = features_dc.reshape(features_dc.shape[0], -1)
    f_rest = features_rest.reshape(features_rest.shape[0], -1)

    names = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(f_dc.shape[1]):
        names.append(f"f_dc_{i}")
    for i in range(f_rest.shape[1]):
        names.append(f"f_rest_{i}")
    names.append("opacity")
    for i in range(scaling.shape[1]):
        names.append(f"scale_{i}")
    for i in range(rotation.shape[1]):
        names.append(f"rot_{i}")

    dtype_full = [(name, "f4") for name in names]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacity, scaling, rotation),
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(plypath)


def get_world_xz_bounds(bbx):
    """
    获取 block bbox 在世界坐标系下的 x/z 范围。

    本项目的分块主要在水平平面上进行，因此内部边界判断主要看 x 和 z：
        x_min, x_max, z_min, z_max
    """
    arr = np.asarray(bbx, dtype=np.float64)
    return arr[:, 0].min(), arr[:, 0].max(), arr[:, 2].min(), arr[:, 2].max()


def find_internal_planes(blocks_info, tol=1e-4):
    """
    自动找出 block 之间的内部边界平面。

    原理：
        如果某个 block 的 x_max 与另一个 block 的 x_min 几乎相等，
        说明这两个 block 在这个 x 位置相邻，这个 x 就是内部边界。

        z 方向同理：
        如果某个 block 的 z_max 与另一个 block 的 z_min 几乎相等，
        说明这个 z 是内部边界。

    参数：
        blocks_info:
            blocks_info.json 读取后的字典，包含每个 block 的 bbox。
        tol:
            判断两个边界是否重合的容差。

    返回：
        internal_x:
            所有 x 方向内部边界位置。
        internal_z:
            所有 z 方向内部边界位置。
    """
    xs_min, xs_max, zs_min, zs_max = [], [], [], []

    for k, v in blocks_info.items():
        if k == "num_blocks":
            continue
        xmin, xmax, zmin, zmax = get_world_xz_bounds(v["bbx"])
        xs_min.append(xmin)
        xs_max.append(xmax)
        zs_min.append(zmin)
        zs_max.append(zmax)

    internal_x = []
    internal_z = []

    for a in xs_max:
        for b in xs_min:
            if abs(a - b) < tol:
                internal_x.append(0.5 * (a + b))

    for a in zs_max:
        for b in zs_min:
            if abs(a - b) < tol:
                internal_z.append(0.5 * (a + b))

    internal_x = sorted(set(round(float(x), 6) for x in internal_x))
    internal_z = sorted(set(round(float(z), 6) for z in internal_z))
    return internal_x, internal_z


def is_close_to_any(value, arr, tol=1e-3):
    """
    判断 value 是否接近 arr 中的任意一个值。

    用途：
        判断某个 bbox 边界是否是内部边界。
        例如：
            当前 block 的 xmin 是否等于某条 internal_x。
    """
    return any(abs(value - a) < tol for a in arr)


def patch_config(src_cfg, dst_cfg, dst_out):
    """
    复制并修改 config.yaml。

    原始 config.yaml 中的 output_dirpath 指向 baseline 输出目录。
    生成 SAOF 结果时，需要把 output_dirpath 改成新的 dst_out，
    否则后续 render_views.py 可能会读错输出路径。
    """
    cfg = yaml.safe_load(src_cfg.read_text())
    cfg["output_dirpath"] = str(dst_out)
    dst_cfg.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True))


def copy_structure(src_out, dst_out):
    """
    复制 BlockGaussian 输出目录中后续评估需要的基础文件。

    会复制：
        blocks_info.json:
            分块边界信息，是 SAOF 判断内部边界的关键文件。
        Partition_Results.png / ROI_region.png:
            可视化文件，方便记录实验。
        config.yaml:
            会同步修改 output_dirpath。
        block_init_pcd:
            初始点云目录，保留目录结构完整性。

    同时会创建：
        point_cloud_postfix/
            用于保存每个 block 经 SAOF 处理后的 ply。
    """
    # An experiment output is immutable once created; never merge into or
    # silently overwrite an earlier parameter candidate.
    dst_out.mkdir(parents=True, exist_ok=False)

    for name in ["blocks_info.json", "Partition_Results.png", "ROI_region.png"]:
        src = src_out / name
        if src.exists():
            shutil.copy2(src, dst_out / name)

    src_cfg = src_out / "config.yaml"
    if not src_cfg.exists():
        raise FileNotFoundError(src_cfg)
    patch_config(src_cfg, dst_out / "config.yaml", dst_out)

    src_block = src_out / "block_init_pcd"
    dst_block = dst_out / "block_init_pcd"
    if src_block.exists() and not dst_block.exists():
        shutil.copytree(src_block, dst_block)

    (dst_out / "point_cloud_postfix").mkdir(exist_ok=True)


def compute_saof_weights(
    xyz_world,
    xyz_block,
    block_info,
    internal_x,
    internal_z,
    overlap_width,
    min_weight,
    mode,
):
    """
    计算 SAOF 的保留范围和 opacity 权重。

    这是本脚本最核心的函数。

    输入：
        xyz_world:
            当前 block 点云中所有 Gaussian 的世界坐标。
        xyz_block:
            当前 block 点云中所有 Gaussian 的 block 局部坐标。
            当前版本的权重主要用世界坐标 x/z 计算，保留该参数是为了接口清晰。
        block_info:
            当前 block 在 blocks_info.json 中的信息，主要使用其中的 bbx。
        internal_x / internal_z:
            所有内部 block 边界的位置。
        overlap_width:
            在内部边界外侧允许额外保留的 overlap 宽度。
        min_weight:
            overlap 最外侧的最低 opacity 权重。
        mode:
            additive:
                原始 bbox 内点的权重保持 1；
                只给跨出 bbox 但仍位于 overlap band 的点降低权重。
                这是当前实验中更稳定的模式。
            ownership:
                bbox 内部靠近边界的点也做平滑权重衰减；
                更接近真正的相邻 block ownership 平滑交接，但风险更高。

    输出：
        keep_xz:
            x/z 平面上是否位于“核心区域 + 内部边界 overlap 区域”。
        core_xz:
            x/z 平面上是否位于原始 bbox 内。
        weight:
            每个 Gaussian 的 SAOF 权重，用于重加权 opacity。

    直观理解：
        原始融合：bbox 外全部丢弃，bbox 内全部保留。
        SAOF 融合：内部边界方向的 bbox 外侧允许保留一小段，
                  但越远离本 block，opacity 权重越低。
    """
    # 当前 block 的原始 bbox 在世界坐标系下的 x/z 范围。
    bbx = np.asarray(block_info["bbx"], dtype=np.float64)
    xmin_w, xmax_w, zmin_w, zmax_w = get_world_xz_bounds(bbx)

    # 判断当前 block 的四条 x/z 边界中，哪些是内部边界。
    # 只有内部边界才允许保留 overlap；场景最外侧边界仍然硬裁剪。
    has_xmin_internal = is_close_to_any(xmin_w, internal_x)
    has_xmax_internal = is_close_to_any(xmax_w, internal_x)
    has_zmin_internal = is_close_to_any(zmin_w, internal_z)
    has_zmax_internal = is_close_to_any(zmax_w, internal_z)

    # 取出每个 Gaussian 的世界坐标 x 和 z。
    # 本脚本主要在水平平面上处理 block 之间的边界过渡。
    xw = xyz_world[:, 0]
    zw = xyz_world[:, 2]

    # 原始 hard crop 的空间范围：
    # 这对应原始 block_fusion.py 的 x/z 裁剪范围。
    core_x = (xw >= xmin_w) & (xw <= xmax_w)
    core_z = (zw >= zmin_w) & (zw <= zmax_w)
    core_xz = core_x & core_z

    # 只在内部边界方向允许向外保留 overlap band；外部场景边界仍硬裁剪。
    keep_x_min = xmin_w - overlap_width if has_xmin_internal else xmin_w
    keep_x_max = xmax_w + overlap_width if has_xmax_internal else xmax_w
    keep_z_min = zmin_w - overlap_width if has_zmin_internal else zmin_w
    keep_z_max = zmax_w + overlap_width if has_zmax_internal else zmax_w

    keep_xz = (
        (xw >= keep_x_min) & (xw <= keep_x_max) &
        (zw >= keep_z_min) & (zw <= keep_z_max)
    )

    # wx / wz 分别记录 x 方向和 z 方向的边界权重。
    # 最终 weight = wx * wz。
    # 对于普通核心区域，默认权重为 1。
    wx = np.ones_like(xw, dtype=np.float32)
    wz = np.ones_like(zw, dtype=np.float32)

    # 防止 overlap_width 为 0 导致除零。
    r = max(float(overlap_width), 1e-6)

    def outside_weight(dist_out):
        """
        bbox 外侧 overlap 区域的权重函数。

        dist_out = 0:
            正好在原始 bbox 边界上，权重为 0.5。
        dist_out = r:
            到达 overlap 最外侧，权重为 min_weight。

        这样可以让跨出本 block 的辅助点逐渐变弱，
        避免直接把邻区点以完整 opacity 合并进去。
        """
        t = np.clip(dist_out / r, 0.0, 1.0)
        return (0.5 * (1.0 - t) + min_weight * t).astype(np.float32)

    def inside_weight(dist_in):
        """
        bbox 内侧靠近边界区域的权重函数，仅在 ownership 模式下使用。

        dist_in = 0:
            正好在原始 bbox 边界上，权重为 0.5。
        dist_in = r:
            进入 block 内部 r 距离后，权重恢复为 1。

        additive 模式不使用这个函数，因为 additive 模式保持 bbox 内部点权重为 1。
        """
        t = np.clip(dist_in / r, 0.0, 1.0)
        return (0.5 * (1.0 - t) + 1.0 * t).astype(np.float32)

    # x-min 侧：
    # 如果当前 block 的 xmin 是内部边界，则允许向 xmin 外侧保留 overlap band。
    if has_xmin_internal:
        m_out = (xw >= xmin_w - r) & (xw < xmin_w)
        wx[m_out] = np.minimum(wx[m_out], outside_weight(xmin_w - xw[m_out]))

        if mode == "ownership":
            m_in = (xw >= xmin_w) & (xw < xmin_w + r)
            wx[m_in] = np.minimum(wx[m_in], inside_weight(xw[m_in] - xmin_w))

    # x-max 侧：
    # 如果当前 block 的 xmax 是内部边界，则允许向 xmax 外侧保留 overlap band。
    if has_xmax_internal:
        m_out = (xw > xmax_w) & (xw <= xmax_w + r)
        wx[m_out] = np.minimum(wx[m_out], outside_weight(xw[m_out] - xmax_w))

        if mode == "ownership":
            m_in = (xw > xmax_w - r) & (xw <= xmax_w)
            wx[m_in] = np.minimum(wx[m_in], inside_weight(xmax_w - xw[m_in]))

    # z-min 侧：
    # 如果当前 block 的 zmin 是内部边界，则允许向 zmin 外侧保留 overlap band。
    if has_zmin_internal:
        m_out = (zw >= zmin_w - r) & (zw < zmin_w)
        wz[m_out] = np.minimum(wz[m_out], outside_weight(zmin_w - zw[m_out]))

        if mode == "ownership":
            m_in = (zw >= zmin_w) & (zw < zmin_w + r)
            wz[m_in] = np.minimum(wz[m_in], inside_weight(zw[m_in] - zmin_w))

    # z-max 侧：
    # 如果当前 block 的 zmax 是内部边界，则允许向 zmax 外侧保留 overlap band。
    if has_zmax_internal:
        m_out = (zw > zmax_w) & (zw <= zmax_w + r)
        wz[m_out] = np.minimum(wz[m_out], outside_weight(zw[m_out] - zmax_w))

        if mode == "ownership":
            m_in = (zw > zmax_w - r) & (zw <= zmax_w)
            wz[m_in] = np.minimum(wz[m_in], inside_weight(zmax_w - zw[m_in]))

    # x 和 z 两个方向的权重相乘。
    # 在十字交界处，一个点可能同时受到 x 边界和 z 边界影响。
    weight = (wx * wz).astype(np.float32)

    return keep_xz, core_xz, weight


def merge_plys(sh_degree, ply_filepaths, merged_plypath):
    """
    将所有 block 处理后的 postfix ply 合并成一个完整场景 ply。

    这里的合并方式仍然是 concatenate，
    但与原始 block_fusion.py 不同的是：
        输入的每个 block ply 已经经过 SAOF 筛选和 opacity 重加权，
        因此最终 point_cloud_merged.ply 是柔性融合后的结果。
    """
    xyz_all, fdc_all, frest_all, op_all, sc_all, rot_all = [], [], [], [], [], []

    for p in ply_filepaths:
        xyz, fdc, frest, op, sc, rot = load_gs_ply(sh_degree, str(p))
        xyz_all.append(xyz)
        fdc_all.append(fdc)
        frest_all.append(frest)
        op_all.append(op)
        sc_all.append(sc)
        rot_all.append(rot)

    xyz_all = np.concatenate(xyz_all, axis=0)
    fdc_all = np.concatenate(fdc_all, axis=0)
    frest_all = np.concatenate(frest_all, axis=0)
    op_all = np.concatenate(op_all, axis=0)
    sc_all = np.concatenate(sc_all, axis=0)
    rot_all = np.concatenate(rot_all, axis=0)

    print("[SAOF] Num points of merged scene pcd:", xyz_all.shape[0])
    save_gs_ply(merged_plypath, xyz_all, fdc_all, frest_all, op_all, sc_all, rot_all)


def process_block(src_ply, dst_ply, block_id, block_info, internal_x, internal_z, segments, args):
    """
    对单个 block 的 point_cloud_60000.ply 执行 SAOF 处理。

    处理流程：
        1. 读取当前 block 的 Gaussian ply；
        2. 计算当前 block 的 bbox 变换；
        3. 保持原始 block_fusion.py 的 y 方向裁剪逻辑；
        4. 在 x/z 方向计算核心区域和 overlap 区域；
        5. 原始 bbox 内点全部保留；
        6. bbox 外 overlap 点按 opacity * weight 过滤；
        7. 对最终保留点的 opacity 做 SAOF 权重重加权；
        8. 保存当前 block 的 postfix ply；
        9. 返回统计信息，方便后续写入 SAOF_SUMMARY.json。
    """
    # 读取当前 block 的所有 Gaussian 属性。
    xyz, fdc, frest, op, sc, rot = load_gs_ply(args.sh_degree, str(src_ply))

    # 将当前 block 的世界坐标 bbox 转换为局部坐标系参数。
    bbx = np.asarray(block_info["bbx"], dtype=np.float32)
    W2B, x_extent, y_extent, z_extent = calculate_block_transform(bbx)

    # 把 Gaussian 中心从世界坐标转换到当前 block 的局部坐标。
    # y 方向裁剪需要用局部坐标，因为 bbox 可能不是简单的轴对齐高度范围。
    xyz_block = (W2B[:3, :3] @ xyz.T).T + W2B[:3, 3]

    # 与原 block_fusion.py 保持一致：y 方向仍用 1.2 倍范围
    y_mask = (xyz_block[:, 1] >= -1.2 * y_extent) & (xyz_block[:, 1] <= 1.2 * y_extent)

    # 计算 x/z 平面上的保留范围和 SAOF 权重。
    keep_xz, core_xz, weight = compute_saof_weights(
        xyz_world=xyz,
        xyz_block=xyz_block,
        block_info=block_info,
        internal_x=internal_x,
        internal_z=internal_z,
        overlap_width=args.overlap_width,
        min_weight=args.min_weight,
        mode=args.mode,
    )

    # hard_core_mask:
    #     原始 bbox 内，并且 y 方向也合法的核心点。
    # candidate_mask:
    #     核心区域 + 内部边界 overlap 区域，并且 y 方向合法的候选点。
    hard_core_mask = core_xz & y_mask
    candidate_mask = keep_xz & y_mask

    # 将 opacity 从 logit 转成概率，方便乘以权重。
    opacity_prob = sigmoid(op.reshape(-1))

    # 原始 bbox 内的点永远保留。
    # bbox 外的 overlap 辅助点只有在 opacity * weight 达到阈值时才保留。
    # 这样可以避免把大量低质量、低贡献的外侧点引入最终融合结果。
    # The global x/z coordinate of a boundary is not enough on irregular
    # partitions: retain an extra point only near an edge actually shared by
    # this block and a neighbour, including the edge's finite endpoint span.
    own_segments = [edge for edge in segments if block_id in edge["blocks"]]
    near_shared_edge, _ = mask_numpy(xyz, own_segments, args.overlap_width)
    extra_mask = candidate_mask & (~hard_core_mask) & near_shared_edge
    extra_keep = extra_mask & ((opacity_prob * weight) >= args.extra_keep_opacity)

    # 最终保留点 = 原始核心点 + 通过阈值筛选的 overlap 辅助点。
    final_mask = hard_core_mask | extra_keep

    # 对最终保留点的 opacity 做重加权。
    # 注意：op 是 logit，因此需要先 sigmoid -> 乘权重 -> logit。
    final_weight = weight[final_mask].reshape(-1, 1)
    op_prob_new = sigmoid(op[final_mask]) * final_weight
    op_new = logit(op_prob_new).astype(op.dtype)

    # 保存当前 block 经 SAOF 处理后的 postfix ply。
    save_gs_ply(
        dst_ply,
        xyz[final_mask],
        fdc[final_mask],
        frest[final_mask],
        op_new,
        sc[final_mask],
        rot[final_mask],
    )

    # 统计 opacity 总量变化，用于观察 SAOF 是否引入了过多/过少贡献。
    hard_opacity_sum = float(opacity_prob[hard_core_mask].sum())
    saof_opacity_sum = float(sigmoid(op_new).sum())

    # 记录当前 block 的处理统计，最后会写入 SAOF_SUMMARY.json。
    stat = {
        "src": str(src_ply),
        "dst": str(dst_ply),
        "input_points": int(xyz.shape[0]),
        "hard_core_points": int(hard_core_mask.sum()),
        "extra_candidates": int(extra_mask.sum()),
        "extra_kept": int(extra_keep.sum()),
        "final_points": int(final_mask.sum()),
        "added_points_vs_hard": int(final_mask.sum() - hard_core_mask.sum()),
        "hard_opacity_sum": hard_opacity_sum,
        "saof_opacity_sum": saof_opacity_sum,
        "saof_to_hard_opacity_sum_ratio": saof_opacity_sum / max(hard_opacity_sum, 1e-12),
        "mean_weight_final": float(final_weight.mean()) if final_weight.size else 1.0,
        "min_weight_final": float(final_weight.min()) if final_weight.size else 1.0,
    }

    return stat


def main():
    """
    命令行入口。

    典型运行方式：

        python -u tools/seam_aware_overlap_fusion.py \
          --src-out output/rubble_b2_60k_3090_labaxis_roi4_bs2 \
          --dst-out output/rubble_b2_60k_4090_labaxis_roi4_bs2_saof_final_w4p5_m002_op0010 \
          --mode additive \
          --overlap-width 4.5 \
          --min-weight 0.02 \
          --extra-keep-opacity 0.010

    当前脚本会从 src-out/blocks_info.json 读取分块信息，
    不需要额外传入 --blocks-info。
    """
    ap = argparse.ArgumentParser()
    # 原始 BlockGaussian 输出目录。
    ap.add_argument("--src-out", required=True)

    # SAOF 结果输出目录。
    ap.add_argument("--dst-out", required=True)

    # 内部边界 overlap band 的宽度。
    ap.add_argument("--overlap-width", type=float, default=8.0)

    # overlap 最外侧的最低 opacity 权重。
    ap.add_argument("--min-weight", type=float, default=0.10)

    # bbox 外 overlap 辅助点的保留阈值。
    # 判断条件为：sigmoid(opacity) * weight >= extra_keep_opacity。
    ap.add_argument("--extra-keep-opacity", type=float, default=0.01)

    # additive：只给 bbox 外 overlap 点降权，bbox 内核心点保持权重 1。
    # ownership：bbox 内靠近边界的点也做平滑 ownership 权重。
    ap.add_argument("--mode", choices=["additive", "ownership"], default="additive")

    # 球谐函数阶数，BlockGaussian 默认通常为 3。
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--iteration", type=int, required=True,
                    help="Per-block training step to read, e.g. 20000")
    args = ap.parse_args()

    if args.overlap_width <= 0 or not 0 <= args.min_weight <= 0.5 or args.extra_keep_opacity < 0:
        ap.error("Require overlap-width > 0, 0 <= min-weight <= 0.5 and extra-keep-opacity >= 0")

    src_out = Path(args.src_out)
    dst_out = Path(args.dst_out)

    # 复制必要目录和配置文件，构建一个完整的新输出目录。
    if src_out.resolve() == dst_out.resolve():
        ap.error("--src-out and --dst-out must differ")
    if dst_out.exists():
        raise FileExistsError(f"Refusing to overwrite existing SAOF output: {dst_out}")
    if not (src_out / "config.yaml").is_file() or not (src_out / "blocks_info.json").is_file():
        raise FileNotFoundError("Source requires config.yaml and blocks_info.json")
    source_info = json.loads((src_out / "blocks_info.json").read_text())
    for block_id in range(source_info["num_blocks"]):
        required_ply = src_out / "point_cloud" / str(block_id) / f"point_cloud_{args.iteration}.ply"
        if not required_ply.is_file():
            raise FileNotFoundError(required_ply)
    copy_structure(src_out, dst_out)

    # 读取分块信息，并自动找出所有内部边界位置。
    blocks_info = json.loads((src_out / "blocks_info.json").read_text())
    internal_x, internal_z = find_internal_planes(blocks_info)
    segments = extract_segments(src_out / "blocks_info.json")

    print("[SAOF] src_out =", src_out)
    print("[SAOF] overlap_width =", args.overlap_width)
    print("[SAOF] min_weight =", args.min_weight)
    print("[SAOF] extra_keep_opacity =", args.extra_keep_opacity)

    # all_stats 用于保存每个 block 的处理统计。
    # ply_filepaths 用于记录每个 block 处理后的 postfix ply，后续统一 merge。
    all_stats = []
    ply_filepaths = []

    # 逐个处理 block。
    # blocks_info 中还有一个 "num_blocks" 字段，需要跳过。
    for k, v in sorted(blocks_info.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 9999):
        if k == "num_blocks":
            continue

        bid = int(k)

        # 当前实验默认读取每个 block 训练 60000 iteration 后的点云。
        # 如果你的 baseline 使用了不同 iteration，需要改这里的文件名。
        src_ply = src_out / "point_cloud" / str(bid) / f"point_cloud_{args.iteration}.ply"

        # 当前 block 经 SAOF 处理后的输出路径。
        dst_ply = dst_out / "point_cloud_postfix" / f"point_cloud_{bid}.ply"

        if not src_ply.exists():
            raise FileNotFoundError(src_ply)

        print(f"[SAOF] processing block {bid}: {src_ply}")
        stat = process_block(src_ply, dst_ply, bid, v, internal_x, internal_z, segments, args)
        all_stats.append(stat)
        ply_filepaths.append(dst_ply)

        print(
            f"[SAOF][block {bid}] "
            f"input={stat['input_points']}, hard={stat['hard_core_points']}, "
            f"extra_candidates={stat['extra_candidates']}, extra_kept={stat['extra_kept']}, "
            f"final={stat['final_points']}, "
            f"opacity_ratio={stat['saof_to_hard_opacity_sum_ratio']:.6f}"
        )

    # 所有 block 都处理完后，把 postfix ply 合并成完整场景 ply。
    merged_plypath = dst_out / "point_cloud_merged.ply"
    merge_plys(args.sh_degree, ply_filepaths, merged_plypath)

    # 保存本次 SAOF 的参数和统计结果，方便后续复现实验或写报告。
    summary = {
        "method": "seam_aware_overlap_fusion",
        "mode": args.mode,
        "src_out": str(src_out),
        "dst_out": str(dst_out),
        "internal_x": internal_x,
        "internal_z": internal_z,
        "finite_boundary_segments": segments,
        "source_iteration": args.iteration,
        "overlap_width": args.overlap_width,
        "min_weight": args.min_weight,
        "extra_keep_opacity": args.extra_keep_opacity,
        "stats": all_stats,
    }

    (dst_out / "SAOF_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[SAOF] saved:", dst_out / "SAOF_SUMMARY.json")
    print("[SAOF] saved:", merged_plypath)
    print("[SAOF] IMPORTANT: do NOT run block_fusion.py on this output, because it will overwrite SAOF postfix results.")


if __name__ == "__main__":
    main()
