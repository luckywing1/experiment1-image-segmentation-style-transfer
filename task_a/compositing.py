"""
task_a/compositing.py
前景提取与图像合成模块
- 羽化掩码边缘（高斯模糊）
- Alpha 融合（软边缘）
- 泊松融合（cv2.seamlessClone，无缝贴合）
- 提取前景 RGBA 图
"""

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────
# 掩码处理
# ─────────────────────────────────────────────────────────────

def feather_mask(mask: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """
    对二值掩码进行高斯模糊羽化，使边缘柔和。

    Args:
        mask:         uint8 (H, W)，前景=255，背景=0
        blur_radius:  高斯核半径，必须为奇数；值越大边缘越柔和

    Returns:
        feathered: float32 (H, W)，值域 [0, 1]，可直接用于 Alpha 融合
    """
    # 确保 blur_radius 为正奇数
    r = max(1, int(blur_radius))
    if r % 2 == 0:
        r += 1
    ksize = (r * 2 + 1, r * 2 + 1)

    blurred = cv2.GaussianBlur(mask.astype(np.float32), ksize, sigmaX=0)
    feathered = np.clip(blurred / 255.0, 0.0, 1.0)
    return feathered


def erode_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    对掩码进行轻度腐蚀，去除边缘噪点。
    与羽化配合使用效果更佳。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    return eroded


# ─────────────────────────────────────────────────────────────
# 前景提取
# ─────────────────────────────────────────────────────────────

def extract_foreground(
    pil_image: Image.Image,
    mask: np.ndarray,
    feather_radius: int = 10,
) -> Image.Image:
    """
    从原图中提取前景，返回带透明通道（RGBA）的 PIL 图像。

    Args:
        pil_image:     原始 PIL RGB 图像
        mask:          uint8 (H, W) 二值掩码，前景=255
        feather_radius: 羽化半径，0 表示不羽化

    Returns:
        fg_rgba: RGBA PIL 图像，背景区域为透明
    """
    img_np = np.array(pil_image.convert("RGB"))  # (H, W, 3)

    if feather_radius > 0:
        alpha = feather_mask(mask, blur_radius=feather_radius)  # float [0,1]
        alpha_uint8 = (alpha * 255).astype(np.uint8)
    else:
        alpha_uint8 = mask.copy()

    # 拼合 RGBA
    rgba = np.dstack([img_np, alpha_uint8])  # (H, W, 4)
    return Image.fromarray(rgba, mode="RGBA")


# ─────────────────────────────────────────────────────────────
# Alpha 融合
# ─────────────────────────────────────────────────────────────

def alpha_blend(
    fg_pil: Image.Image,
    bg_pil: Image.Image,
    mask: np.ndarray,
    feather_radius: int = 10,
    position: tuple = (0, 0),
    scale: float = 1.0,
) -> Image.Image:
    """
    将前景 Alpha 融合到背景中。

    Args:
        fg_pil:        前景 PIL RGB 图像（原始尺寸）
        bg_pil:        背景 PIL RGB 图像
        mask:          uint8 (H, W) 二值掩码
        feather_radius: 羽化半径
        position:      (x, y) 前景左上角在背景中的位置
        scale:         前景缩放比例

    Returns:
        result: RGB PIL 图像（融合结果）
    """
    bg = np.array(bg_pil.convert("RGB")).astype(np.float32)
    fg_np = np.array(fg_pil.convert("RGB")).astype(np.float32)

    # 获取 alpha 通道 [0,1]
    if feather_radius > 0:
        alpha = feather_mask(mask, blur_radius=feather_radius)
    else:
        alpha = (mask / 255.0).astype(np.float32)

    # 缩放前景和掩码
    if scale != 1.0:
        new_h = int(fg_np.shape[0] * scale)
        new_w = int(fg_np.shape[1] * scale)
        fg_np = cv2.resize(fg_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算粘贴区域（支持边界裁剪）
    bh, bw = bg.shape[:2]
    fh, fw = fg_np.shape[:2]
    x, y = int(position[0]), int(position[1])

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + fw, bw), min(y + fh, bh)

    # 对应的前景裁剪区域
    fx1 = x1 - x
    fy1 = y1 - y
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return bg_pil  # 没有重叠区域

    fg_crop = fg_np[fy1:fy2, fx1:fx2]
    a_crop = alpha[fy1:fy2, fx1:fx2, np.newaxis]  # (h, w, 1)

    bg[y1:y2, x1:x2] = fg_crop * a_crop + bg[y1:y2, x1:x2] * (1.0 - a_crop)

    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────
# 泊松融合
# ─────────────────────────────────────────────────────────────

def poisson_blend(
    fg_pil: Image.Image,
    bg_pil: Image.Image,
    mask: np.ndarray,
    position: tuple = (0, 0),
    scale: float = 1.0,
) -> Image.Image:
    """
    使用 OpenCV seamlessClone 实现泊松融合（无缝合成）。
    适合前景与背景光照差异较大的场景。

    Args:
        fg_pil:    前景 PIL RGB 图像（与 mask 对应）
        bg_pil:    背景 PIL RGB 图像
        mask:      uint8 (H, W) 二值掩码（无需羽化）
        position:  (x, y) 前景中心在背景中的位置（seamlessClone 使用中心点）
        scale:     前景缩放比例

    Returns:
        result: RGB PIL 图像
    """
    fg_bgr = cv2.cvtColor(np.array(fg_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    bg_bgr = cv2.cvtColor(np.array(bg_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask_u8 = mask.copy()

    # 缩放
    if scale != 1.0:
        new_h = int(fg_bgr.shape[0] * scale)
        new_w = int(fg_bgr.shape[1] * scale)
        fg_bgr = cv2.resize(fg_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_u8 = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # seamlessClone 需要前景尺寸不超过背景
    bh, bw = bg_bgr.shape[:2]
    fh, fw = fg_bgr.shape[:2]

    # 中心点：将前景中心放到 position
    cx = int(position[0]) + fw // 2
    cy = int(position[1]) + fh // 2
    cx = np.clip(cx, fw // 2, bw - fw // 2 - 1)
    cy = np.clip(cy, fh // 2, bh - fh // 2 - 1)

    # 调整前景尺寸适配背景（如超出则 fallback 到 Alpha 融合）
    if fw >= bw or fh >= bh:
        print("[poisson_blend] 前景尺寸超过背景，回退到 Alpha 融合")
        return alpha_blend(fg_pil, bg_pil, mask, feather_radius=10, position=position, scale=scale)

    try:
        result_bgr = cv2.seamlessClone(
            fg_bgr, bg_bgr, mask_u8,
            (cx, cy),
            cv2.NORMAL_CLONE
        )
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    except cv2.error as e:
        print(f"[poisson_blend] seamlessClone 失败：{e}，回退到 Alpha 融合")
        return alpha_blend(fg_pil, bg_pil, mask, feather_radius=10, position=position, scale=scale)


# ─────────────────────────────────────────────────────────────
# 可视化辅助
# ─────────────────────────────────────────────────────────────

def mask_to_display(mask: np.ndarray) -> Image.Image:
    """将二值掩码转换为灰度 PIL 图像，用于界面预览。"""
    return Image.fromarray(mask, mode="L").convert("RGB")


def make_checkerboard_bg(size: tuple, tile: int = 20) -> Image.Image:
    """
    生成棋盘格背景（用于展示透明区域）。

    Args:
        size: (width, height)
        tile: 格子大小（像素）

    Returns:
        PIL RGB 图像
    """
    w, h = size
    board = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if (x // tile + y // tile) % 2 == 0:
                board[y:y+tile, x:x+tile] = 200
            else:
                board[y:y+tile, x:x+tile] = 255
    return Image.fromarray(board)


def overlay_rgba_on_checker(fg_rgba: Image.Image) -> Image.Image:
    """
    将 RGBA 前景叠加到棋盘格背景上，直观展示透明度。

    Args:
        fg_rgba: RGBA PIL 图像

    Returns:
        RGB PIL 图像
    """
    checker = make_checkerboard_bg(fg_rgba.size)
    checker_rgba = checker.convert("RGBA")
    merged = Image.alpha_composite(checker_rgba, fg_rgba.convert("RGBA"))
    return merged.convert("RGB")
