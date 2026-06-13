"""
task_b/model/transform.py
图像预处理与后处理工具
- PIL / numpy / torch 之间的转换
- 图像尺寸调整策略
"""

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────
# PIL ↔ Tensor
# ─────────────────────────────────────────────────────────────

def pil_to_tensor(pil_img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """
    将 PIL RGB 图像转换为 tensor，形状 (1, 3, H, W)。
    仅使用 ToTensor（[0, 1] 范围），不做 VGG 归一化。
    这与原始 pytorch-AdaIN 的预处理方式一致。

    Args:
        pil_img: PIL RGB 图像
        device:  目标设备

    Returns:
        tensor: (1, 3, H, W) float32，值域 [0, 1]
    """
    img_rgb = pil_img.convert("RGB")
    t = T.ToTensor()(img_rgb)        # (3, H, W)，值域 [0, 1]
    return t.unsqueeze(0).to(device)  # (1, 3, H, W)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将网络输出 tensor 转为 PIL RGB 图像。
    解码器输出已在 [0, 1] 范围，只需 clamp 后转换。
    不需要反归一化（与原始 pytorch-AdaIN 一致）。

    Args:
        tensor: (1, 3, H, W) 或 (3, H, W) float32

    Returns:
        PIL RGB 图像
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)   # → (3, H, W)

    t = tensor.cpu().detach()
    # 直接 clamp 到 [0, 1] 并转 uint8（无需反归一化）
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (H, W, 3)
    return Image.fromarray(arr)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """uint8 (H, W, 3) numpy → PIL RGB"""
    return Image.fromarray(arr.astype(np.uint8))


def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB → uint8 (H, W, 3) numpy"""
    return np.array(pil_img.convert("RGB"))


# ─────────────────────────────────────────────────────────────
# 尺寸调整
# ─────────────────────────────────────────────────────────────

def resize_keep_ratio(pil_img: Image.Image, max_size: int = 512) -> Image.Image:
    """
    等比缩放，确保最长边不超过 max_size。

    Args:
        pil_img:  PIL 图像
        max_size: 最长边上限（像素）

    Returns:
        缩放后的 PIL 图像
    """
    w, h = pil_img.size
    if max(w, h) <= max_size:
        return pil_img
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def resize_to_match(pil_img: Image.Image, target: Image.Image) -> Image.Image:
    """将 pil_img 缩放到与 target 相同尺寸。"""
    return pil_img.resize(target.size, Image.LANCZOS)


def crop_center(pil_img: Image.Image, crop_w: int, crop_h: int) -> Image.Image:
    """中心裁剪。"""
    w, h = pil_img.size
    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    return pil_img.crop((x, y, x + crop_w, y + crop_h))


def pad_to_multiple(pil_img: Image.Image, multiple: int = 32) -> tuple:
    """
    将图像 padding 到 multiple 的整数倍（避免网络下采样错误）。

    Returns:
        (padded_img, (orig_w, orig_h)) 原始尺寸用于还原
    """
    orig_w, orig_h = pil_img.size
    pad_w = (multiple - orig_w % multiple) % multiple
    pad_h = (multiple - orig_h % multiple) % multiple
    if pad_w == 0 and pad_h == 0:
        return pil_img, (orig_w, orig_h)

    import cv2
    arr = np.array(pil_img)
    arr_padded = cv2.copyMakeBorder(arr, 0, pad_h, 0, pad_w,
                                    cv2.BORDER_REFLECT_101)
    return Image.fromarray(arr_padded), (orig_w, orig_h)


def unpad(pil_img: Image.Image, orig_size: tuple) -> Image.Image:
    """去除 pad，还原为原始尺寸。"""
    return pil_img.crop((0, 0, orig_size[0], orig_size[1]))


# ─────────────────────────────────────────────────────────────
# 帧间一致性（时序平滑）
# ─────────────────────────────────────────────────────────────

class TemporalSmoother:
    """
    简单的指数移动平均（EMA）帧间一致性。
    在视频/摄像头模式下减少帧间闪烁。

    """

    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: 当前帧权重 [0, 1]；越大越接近当前帧（闪烁越明显），
                   越小越平滑（但运动拖影越严重）
        """
        self.alpha = alpha
        self._prev: np.ndarray = None

    def update(self, current: Image.Image) -> Image.Image:
        """
        输入当前帧，返回平滑后的帧。

        Args:
            current: PIL RGB 图像

        Returns:
            smoothed: PIL RGB 图像
        """
        cur_np = np.array(current).astype(np.float32)

        if self._prev is None:
            self._prev = cur_np
            return current

        smoothed = self.alpha * cur_np + (1.0 - self.alpha) * self._prev
        self._prev = smoothed
        return Image.fromarray(np.clip(smoothed, 0, 255).astype(np.uint8))

    def reset(self):
        """重置历史帧（切换场景或重新开始时调用）。"""
        self._prev = None
