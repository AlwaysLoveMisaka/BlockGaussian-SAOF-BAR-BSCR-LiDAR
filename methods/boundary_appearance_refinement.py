"""
saof_v2a_boundary_refine.py

用途：
    该脚本对应项目中的“边界外观精修”步骤。

    它通常接在“边界重叠柔性融合”之后使用：
        baseline / SAOF 融合结果
            ↓
        本脚本：只对 block 内部边界附近的 Gaussian 做短程再优化
            ↓
        输出新的 point_cloud_merged.ply

核心思想：
    BlockGaussian 原始流程是多个 block 独立训练，再进行融合。
    即使融合阶段已经做了重叠区柔性处理，block 边界附近仍可能存在：
        1. 颜色略不一致；
        2. 透明度略不协调；
        3. 相邻 block 在边界处没有经过联合校准。

    因此，本脚本不是重新训练整个场景，也不是移动所有 Gaussian，
    而是只找出 internal block boundary 附近的一小部分 Gaussian，
    冻结非边界区域，并只优化边界 Gaussian 的：
        - features_dc：基础颜色 / 低阶颜色项；
        - opacity：透明度。

    默认不优化：
        - xyz：位置；
        - scaling：尺度；
        - rotation：旋转；
        - features_rest：高阶球谐颜色项。

    这样做的好处是：
        1. 不破坏主体几何结构；
        2. 只修最容易出问题的边界区域；
        3. 训练成本远低于全场景重新训练；
        4. 更适合作为融合后的轻量级边界校准步骤。

典型调用：
    python -u tools/saof_v2a_boundary_refine.py \
      --src-out output/rubble_b2_60k_4090_labaxis_roi4_bs2_saof_final_w4p5_m002_op0010 \
      --dst-out output/saof_v2a_step_sweep_w6_lr02_round2/w6_lr02_s1600 \
      --blocks-info output/rubble_b2_60k_3090_labaxis_roi4_bs2/blocks_info.json \
      --boundary-width 6.0 \
      --steps 1600 \
      --lr-scale 0.2

重要参数：
    --src-out:
        输入模型目录，要求里面有 point_cloud_merged.ply 和 config.yaml。

    --dst-out:
        输出模型目录。若目录已存在，脚本会拒绝覆盖。

    --blocks-info:
        baseline 分块信息文件 blocks_info.json，用于确定 internal boundary 的位置。

    --boundary-width:
        边界带宽度。距离内部 block 边界不超过该宽度的 Gaussian 会被认为是边界 Gaussian。

    --steps:
        边界精修步数。

    --lr-scale:
        学习率缩放系数。实际学习率来自 config.yaml 中的 feature_lr / opacity_lr，
        再乘以该系数。

    --opt-rest-sh:
        默认不优化 features_rest。如果加上该参数，则同时优化高阶 SH 颜色项。
        一般默认关闭更稳。
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 1. 设置项目根目录
# -----------------------------------------------------------------------------
# 本脚本通常放在 BlockGaussian/tools 目录下。
# 由于需要 import 项目内部模块，例如 gaussian_renderer、scene、utils 等，
# 这里把项目根目录加入 sys.path，保证脚本无论从哪里启动都能找到这些模块。
PROJECT_ROOT = Path(os.environ.get("BLOCKGAUSSIAN_ROOT", Path(__file__).resolve().parents[1]))
if not PROJECT_ROOT.is_dir():
    raise FileNotFoundError(f"BlockGaussian repository not found: {PROJECT_ROOT}; set BLOCKGAUSSIAN_ROOT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# 2. 导入 BlockGaussian / 3DGS 相关模块
# -----------------------------------------------------------------------------
# render:
#     给定相机和 Gaussian 模型，渲染出当前视角图像。
#
# get_render_camera:
#     根据图像尺寸、外参、内参构造渲染相机。
#
# GaussianModel:
#     3D Gaussian 模型类，包含 xyz、opacity、features_dc、features_rest、
#     scaling、rotation 等参数。
#
# Scene / SceneDataset:
#     负责读取训练视角、图像、相机参数等数据。
#
# l1_loss / ssim:
#     图像重建损失。这里仍沿用 3DGS / BlockGaussian 常见的
#     L1 + DSSIM 组合损失。
#
# parse_cfg:
#     读取并解析 config.yaml。
from gaussian_renderer import render
from scene.cameras import get_render_camera
from scene.gaussian_model import GaussianModel
from scene.scene_loader import Scene, SceneDataset
from utils.loss_utils import l1_loss, ssim
from utils.utils import parse_cfg
from boundary_segments import extract_segments, mask_torch


def get_internal_planes(blocks_info_path):
    """
    从 blocks_info.json 中提取 internal block boundary 的 x / z 坐标。

    BlockGaussian 的分块主要在 x-z 平面上进行。
    每个 block 的 bbx 都有一个 x 范围和 z 范围，例如：
        x_min, x_max
        z_min, z_max

    如果两个相邻 block 的边界重合，例如：
        block A 的 x_max == block B 的 x_min
    那么这个 x 坐标就是一个 internal boundary。

    这里的实现方式：
        1. 收集所有 block 的 x_min / x_max / z_min / z_max；
        2. 排序并去重；
        3. 去掉最外侧边界；
        4. 剩下的中间边界就是 internal boundary。

    为什么去掉最外侧？
        最外侧是整个场景的外边界，不是两个 block 的交界。
        本脚本只关心 block 与 block 之间的内部边界。

    参数：
        blocks_info_path:
            blocks_info.json 路径。

    返回：
        internal_x:
            所有内部 x 分割面坐标。
        internal_z:
            所有内部 z 分割面坐标。
    """
    with open(blocks_info_path, "r") as f:
        info = json.load(f)

    xs, zs = [], []

    # 遍历每个 block，收集 bbx 的 x / z 范围端点。
    for k, v in info.items():
        if k == "num_blocks":
            continue

        bbx = np.asarray(v["bbx"], dtype=np.float64)

        # bbx 是 8 个顶点，直接取 x 和 z 的最小/最大值即可得到该 block 的平面范围。
        xs += [float(bbx[:, 0].min()), float(bbx[:, 0].max())]
        zs += [float(bbx[:, 2].min()), float(bbx[:, 2].max())]

    def unique_sorted(vals):
        """
        对边界坐标排序并去重。

        使用 1e-4 容差，是因为浮点坐标可能存在极小误差。
        """
        vals = sorted(vals)
        out = []
        for x in vals:
            if not out or abs(x - out[-1]) > 1e-4:
                out.append(x)
        return out

    ux = unique_sorted(xs)
    uz = unique_sorted(zs)

    # 去掉最外侧边界，只保留内部 block 分割面。
    internal_x = ux[1:-1] if len(ux) > 2 else []
    internal_z = uz[1:-1] if len(uz) > 2 else []

    return internal_x, internal_z


def make_boundary_mask(xyz, internal_x, internal_z, width):
    """
    根据 internal boundary 位置，为所有 Gaussian 生成边界 mask。

    判断逻辑：
        对每个 Gaussian，计算它到所有内部 x 边界、z 边界的最小距离。
        如果最小距离 <= boundary_width，则认为它属于边界区域。

    注意：
        这里是在世界坐标中直接判断 x / z 距离。
        因为 BlockGaussian 的分块边界已经在世界坐标 x-z 平面上定义。

    参数：
        xyz:
            Gaussian 的三维坐标，形状一般是 [N, 3]，torch.Tensor。

        internal_x:
            内部 x 边界坐标列表。

        internal_z:
            内部 z 边界坐标列表。

        width:
            边界带宽度。距离内部边界小于等于 width 的点会被选中。

    返回：
        boundary_mask:
            bool Tensor，形状 [N]。
            True 表示该 Gaussian 是边界 Gaussian，允许更新；
            False 表示该 Gaussian 是非边界 Gaussian，不允许更新。

        dist:
            每个 Gaussian 到最近 internal boundary 的距离。
            这个值主要用于统计或调试。
    """
    # 初始化为无穷大，后续不断取到各个边界的最小距离。
    dist = torch.full(
        (xyz.shape[0],),
        float("inf"),
        device=xyz.device,
        dtype=xyz.dtype,
    )

    # 计算到每个内部 x 分割面的距离。
    for px in internal_x:
        dist = torch.minimum(dist, torch.abs(xyz[:, 0] - float(px)))

    # 计算到每个内部 z 分割面的距离。
    for pz in internal_z:
        dist = torch.minimum(dist, torch.abs(xyz[:, 2] - float(pz)))

    return dist <= float(width), dist


def masked_grad(param, mask, dims):
    """
    根据 boundary_mask 屏蔽梯度，只允许边界 Gaussian 的参数被更新。

    这是本脚本的关键函数之一。

    背景：
        即使我们只把 features_dc / opacity 放进 optimizer，
        默认情况下所有 Gaussian 的这些参数都会有梯度。
        但我们希望只更新边界区域的 Gaussian，
        非边界区域应该完全冻结。

    做法：
        在 loss.backward() 之后、optimizer.step() 之前，
        将非边界 Gaussian 的梯度乘以 0。

    参数：
        param:
            需要屏蔽梯度的参数，例如：
                gaussians._features_dc
                gaussians._opacity
                gaussians._features_rest

        mask:
            shape 为 [N] 的 bool Tensor。
            True 表示保留梯度；
            False 表示梯度置零。

        dims:
            参数相对于 [N] 之后还有多少个维度，需要用于扩展 mask。

            例如：
                opacity 形状通常是 [N, 1]，dims=1。
                features_dc 形状通常是 [N, 1, 3]，dims=2。
                features_rest 形状通常是 [N, K, 3]，dims=2。

    例子：
        mask: [N]
        features_dc: [N, 1, 3]
        需要把 mask 扩展成 [N, 1, 1] 才能逐 Gaussian 相乘。
    """
    if param is None or param.grad is None:
        return

    m = mask
    for _ in range(dims):
        m = m.unsqueeze(-1)

    # 非边界点 mask=False，梯度乘以 0；
    # 边界点 mask=True，梯度保持不变。
    param.grad.mul_(m)


def main():
    """
    主流程：

        1. 读取参数；
        2. 复制 config 和必要文件到输出目录；
        3. 读取融合后的 point_cloud_merged.ply；
        4. 根据 blocks_info.json 找到 internal boundary；
        5. 构造 boundary_mask；
        6. 冻结几何参数，只开放颜色和透明度；
        7. 遍历训练视角进行短程优化；
        8. 每一步反向传播后屏蔽非边界 Gaussian 梯度；
        9. 保存新的 point_cloud_merged.ply 和 summary 文件。
    """
    ap = argparse.ArgumentParser()

    # 输入模型目录，通常是 SAOF-v1 输出目录，里面必须有：
    #   point_cloud_merged.ply
    #   config.yaml
    ap.add_argument("--src-out", required=True)

    # 输出目录，本脚本会重新创建该目录。
    ap.add_argument("--dst-out", required=True)

    # blocks_info.json 路径，用于确定内部 block 边界位置。
    ap.add_argument("--blocks-info", required=True)

    # 边界带宽度。最终有效实验中常用 6.0。
    ap.add_argument("--boundary-width", type=float, default=6.0)

    # 精修步数。最终有效实验中 v2a 使用过 1600。
    ap.add_argument("--steps", type=int, default=1500)

    # 学习率缩放系数。
    # 实际 features_dc 学习率 = cfg.feature_lr * lr_scale。
    # 实际 opacity 学习率 = cfg.opacity_lr * lr_scale。
    ap.add_argument("--lr-scale", type=float, default=0.2)

    # 随机种子，保证 shuffle 和训练过程尽量可复现。
    ap.add_argument("--seed", type=int, default=42)

    # 是否优化 features_rest。
    # 默认关闭，因为只优化 DC color + opacity 更稳。
    ap.add_argument("--opt-rest-sh", action="store_true")

    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # 固定随机种子
    # -------------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    src_out = Path(args.src_out)
    dst_out = Path(args.dst_out)
    if src_out.resolve() == dst_out.resolve():
        ap.error("--src-out and --dst-out must differ")
    if dst_out.exists():
        raise FileExistsError(f"Refusing to overwrite existing BAR output: {dst_out}")
    if args.boundary_width <= 0 or args.steps <= 0 or args.lr_scale <= 0:
        ap.error("boundary-width, steps and lr-scale must be positive")

    src_ply = src_out / "point_cloud_merged.ply"
    src_cfg = src_out / "config.yaml"

    # 检查输入文件是否存在。
    if not src_ply.exists():
        raise FileNotFoundError(src_ply)
    if not src_cfg.exists():
        raise FileNotFoundError(src_cfg)

    # -------------------------------------------------------------------------
    # 创建输出目录
    # -------------------------------------------------------------------------
    # Never remove a prior parameter candidate or baseline output.
    dst_out.mkdir(parents=True, exist_ok=False)

    # -------------------------------------------------------------------------
    # 复制并修改 config.yaml
    # -------------------------------------------------------------------------
    # 输出目录中的 config.yaml 应该指向新的 output_dirpath，
    # 否则后续 render_views.py / metrics.py 可能会读错路径。
    cfg_data = yaml.safe_load(src_cfg.read_text())
    cfg_data["output_dirpath"] = str(dst_out)

    dst_cfg = dst_out / "config.yaml"
    dst_cfg.write_text(yaml.safe_dump(cfg_data, sort_keys=False, allow_unicode=True))

    # 复制一些辅助文件，便于后续记录和复现。
    for name in ["blocks_info.json", "SAOF_SUMMARY.json"]:
        p = src_out / name
        if p.exists():
            shutil.copy2(p, dst_out / name)

    # -------------------------------------------------------------------------
    # 解析配置
    # -------------------------------------------------------------------------
    # parse_cfg 是项目原有工具函数，会把 config.yaml 转成可访问属性的 cfg。
    cfg_args = argparse.Namespace(
        config=str(dst_cfg),
        scene_dirpath=None,
        output_dirpath=None,
    )
    cfg = parse_cfg(cfg_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 加载融合后的 Gaussian 模型
    # -------------------------------------------------------------------------
    print("[SAOF-v2a] loading Gaussian:", src_ply)

    sh_degree = getattr(cfg, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(src_ply))

    # -------------------------------------------------------------------------
    # 冻结 / 开放参数
    # -------------------------------------------------------------------------
    # 本脚本的默认设计是“只修外观，不修几何”。
    # 因此冻结：
    #   _xyz      位置
    #   _scaling  尺度
    #   _rotation 旋转
    #
    # 开放：
    #   _features_dc 基础颜色
    #   _opacity     透明度
    #
    # 可选开放：
    #   _features_rest 高阶颜色项
    for p in [
        gaussians._xyz,
        gaussians._scaling,
        gaussians._rotation,
    ]:
        p.requires_grad_(False)

    gaussians._features_dc.requires_grad_(True)
    gaussians._opacity.requires_grad_(True)

    if args.opt_rest_sh:
        gaussians._features_rest.requires_grad_(True)
    else:
        gaussians._features_rest.requires_grad_(False)

    # -------------------------------------------------------------------------
    # 生成边界 mask
    # -------------------------------------------------------------------------
    # internal_x / internal_z 是内部 block 边界坐标。
    # boundary_mask=True 的 Gaussian 才允许更新。
    segments = extract_segments(args.blocks_info)
    internal_x = sorted({edge["coordinate"] for edge in segments if edge["axis"] == "x"})
    internal_z = sorted({edge["coordinate"] for edge in segments if edge["axis"] == "z"})
    boundary_mask, boundary_dist = mask_torch(gaussians.get_xyz, segments, args.boundary_width)

    n_total = int(boundary_mask.numel())
    n_boundary = int(boundary_mask.sum().item())
    if n_boundary == 0:
        raise ValueError("No Gaussians lie inside finite boundary segments at this width")

    print("[SAOF-v2a] internal_x:", internal_x)
    print("[SAOF-v2a] internal_z:", internal_z)
    print("[SAOF-v2a] boundary_width:", args.boundary_width)
    print("[SAOF-v2a] total points:", n_total)
    print("[SAOF-v2a] boundary points:", n_boundary)
    print("[SAOF-v2a] boundary ratio:", n_boundary / max(n_total, 1))

    # -------------------------------------------------------------------------
    # 构造优化器
    # -------------------------------------------------------------------------
    # 学习率继承 config.yaml 中原始 3DGS / BlockGaussian 的学习率，
    # 再乘以 lr_scale。
    #
    # 这样做的好处是：
    #   1. 和原训练配置保持一致；
    #   2. 可以用 lr_scale 控制精修强度；
    #   3. 避免手写绝对学习率导致不稳定。
    params = [
        {
            "params": [gaussians._features_dc],
            "lr": float(getattr(cfg, "feature_lr", 0.0025)) * args.lr_scale,
            "name": "features_dc",
        },
        {
            "params": [gaussians._opacity],
            "lr": float(getattr(cfg, "opacity_lr", 0.05)) * args.lr_scale,
            "name": "opacity",
        },
    ]

    # 如果开启 opt-rest-sh，则给 features_rest 一个更小的学习率。
    # 高阶 SH 更容易引入不稳定，因此这里再乘 0.5。
    if args.opt_rest_sh:
        params.append({
            "params": [gaussians._features_rest],
            "lr": float(getattr(cfg, "feature_lr", 0.0025)) * args.lr_scale * 0.5,
            "name": "features_rest",
        })

    optimizer = torch.optim.Adam(params, eps=1e-15)

    print("[SAOF-v2a] optimizer param groups:")
    for g in optimizer.param_groups:
        print("  ", g["name"], "lr=", g["lr"])

    # -------------------------------------------------------------------------
    # 加载训练场景和训练视角
    # -------------------------------------------------------------------------
    print("[SAOF-v2a] loading scene")
    scene = Scene(
        cfg.scene_dirpath,
        evaluate=cfg.evaluate,
        scene_scale=cfg.scene_scale,
    )

    train_ids = list(scene.train_views_id)
    views_info = [scene.views_info[i] for i in train_ids]

    dataset = SceneDataset(
        views_info,
        cfg.image_scale,
        cfg.scene_scale,
        len(views_info),
        preload=False,
    )

    # batch_size=1 是为了简单稳定，同时降低显存压力。
    # shuffle=True 表示每轮随机取训练视角。
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # 渲染背景颜色，与原 config 保持一致。
    white_bg = bool(getattr(cfg, "white_background", False))
    bg = torch.tensor(
        [1, 1, 1] if white_bg else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )

    # L = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
    # 这里沿用原始 3DGS / BlockGaussian 的图像重建损失形式。
    lambda_dssim = float(getattr(cfg, "lambda_dssim", 0.2))

    # -------------------------------------------------------------------------
    # 开始边界精修
    # -------------------------------------------------------------------------
    print("[SAOF-v2a] start refinement")
    data_iter = iter(loader)

    loss_ema = None

    pbar = tqdm(range(1, args.steps + 1), desc="SAOF-v2a")
    for step in pbar:
        # DataLoader 遍历完后重新开始。
        # 这样 steps 可以大于一个 epoch 的长度。
        try:
            view_info = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            view_info = next(data_iter)

        # 读取当前训练视角的相机参数和 GT 图像。
        extrinsic = view_info["extrinsic"][0].to(device)
        intrinsic = view_info["intrinsic"][0].to(device)
        image_gt = view_info["image"][0].to(device)

        image_height = int(view_info["image_height"][0].item())
        image_width = int(view_info["image_width"][0].item())

        camera = get_render_camera(
            image_height,
            image_width,
            extrinsic,
            intrinsic,
        )

        # 当前完整场景渲染结果。
        render_pkg = render(camera, gaussians, cfg, bg)
        image = render_pkg["render"]

        # 计算图像重建损失。
        Ll1 = l1_loss(image, image_gt)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, image_gt))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # ---------------------------------------------------------------------
        # 关键步骤：屏蔽非边界 Gaussian 的梯度
        # ---------------------------------------------------------------------
        # 虽然渲染和 loss 是基于完整场景计算的，
        # 但真正允许更新的只有 boundary_mask=True 的 Gaussian。
        #
        # 这保证了：
        #   1. 非边界区域作为稳定背景参与渲染；
        #   2. 误差反传时只修边界区域；
        #   3. 全局几何和主体外观不会被破坏。
        masked_grad(gaussians._features_dc, boundary_mask, dims=2)
        masked_grad(gaussians._opacity, boundary_mask, dims=1)

        if args.opt_rest_sh:
            masked_grad(gaussians._features_rest, boundary_mask, dims=2)

        optimizer.step()

        # 限制 opacity 的 logit 范围，避免数值过大或过小导致训练不稳定。
        with torch.no_grad():
            gaussians._opacity.data.clamp_(min=-20.0, max=20.0)

        # 记录 EMA loss，便于观察训练趋势。
        loss_val = float(loss.detach().item())
        loss_ema = loss_val if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_val

        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss_ema:.5f}")

    # -------------------------------------------------------------------------
    # 保存精修后的完整场景
    # -------------------------------------------------------------------------
    out_ply = dst_out / "point_cloud_merged.ply"
    gaussians.save_ply(str(out_ply))

    # 记录本次运行的关键参数和统计信息，便于复现实验。
    summary = {
        "method": "SAOF-v2a boundary-only opacity/features_dc refinement",
        "src_out": str(src_out),
        "dst_out": str(dst_out),
        "blocks_info": str(args.blocks_info),
        "boundary_width": args.boundary_width,
        "steps": args.steps,
        "lr_scale": args.lr_scale,
        "opt_rest_sh": bool(args.opt_rest_sh),
        "total_points": n_total,
        "boundary_points": n_boundary,
        "boundary_ratio": n_boundary / max(n_total, 1),
        "internal_x": internal_x,
        "internal_z": internal_z,
        "finite_boundary_segments": segments,
        "final_loss_ema": loss_ema,
    }

    (dst_out / "SAOF_V2A_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print("[SAOF-v2a] saved:", out_ply)
    print("[SAOF-v2a] saved:", dst_out / "SAOF_V2A_SUMMARY.json")
    print("[SAOF-v2a] done")


if __name__ == "__main__":
    main()
