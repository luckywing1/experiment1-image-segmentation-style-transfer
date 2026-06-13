"""
task_b/style_transfer.py
AdaIN 风格迁移推理引擎
- 单张图像迁移（毫秒级）
- 批量处理文件夹
- 摄像头/视频帧迁移（含帧间一致性）
- 推理耗时记录
"""

from task_b.model.transform import (
    pil_to_tensor,
    tensor_to_pil,
    resize_keep_ratio,
    TemporalSmoother,
)
from task_b.model.adain import AdaINNet, DEFAULT_VGG_NORMALISED_PATH, DEFAULT_DECODER_PATH as _ADAIN_DEFAULT_DECODER
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import cv2
import numpy as np
from PIL import Image

# 项目根目录
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# 支持的图像扩展名
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 默认解码器权重路径
DEFAULT_DECODER_PATH = _ADAIN_DEFAULT_DECODER

# 默认 VGG 编码器权重路径
DEFAULT_VGG_PATH = DEFAULT_VGG_NORMALISED_PATH


class StyleTransfer:
    """
    AdaIN 风格迁移推理引擎。

    """

    def __init__(self, device: str = None, max_content_size: int = 512,
                 max_style_size: int = 512):
        """
        Args:
            device:           推理设备（None 自动选择）
            max_content_size: 内容图最长边上限（像素），影响推理速度与显存
            max_style_size:   风格图最长边上限
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_content_size = max_content_size
        self.max_style_size = max_style_size

        self._net: Optional[AdaINNet] = None
        self._loaded = False

        # 帧间平滑器（视频模式）
        self._smoother = TemporalSmoother(alpha=0.7)

        # 统计信息
        self.last_inference_ms: float = 0.0
        self.last_total_ms: float = 0.0

    def load(self, decoder_path: str = DEFAULT_DECODER_PATH,
             vgg_normalised_path: str = DEFAULT_VGG_PATH):
        """
        加载模型。必须在 transfer() 前调用。

        权重文件不存在时会自动从 GitHub 下载。

        Args:
            decoder_path:        decoder.pth 文件路径
            vgg_normalised_path: vgg_normalised.pth 文件路径
        """
        # 自动下载缺失的权重文件
        if not os.path.exists(decoder_path):
            print(f"[StyleTransfer] decoder.pth 不存在，正在自动下载...")
            from task_b.model.adain import download_decoder
            download_decoder(save_path=decoder_path)

        if not os.path.exists(vgg_normalised_path):
            print(f"[StyleTransfer] vgg_normalised.pth 不存在，正在自动下载...")
            from task_b.model.adain import download_vgg_normalised
            download_vgg_normalised(save_path=vgg_normalised_path)

        print(f"[StyleTransfer] 加载编码器（VGG19 normalised）...")
        self._net = AdaINNet(vgg_normalised_path=vgg_normalised_path)
        self._net.load_decoder(decoder_path)
        self._net.to(self.device)
        self._net.eval()
        self._loaded = True
        print(f"[StyleTransfer] 模型就绪，设备：{self.device}")

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("请先调用 load() 加载模型！")

    def transfer(
        self,
        content: Image.Image,
        style: Image.Image,
        alpha: float = 1.0,
        temporal_smooth: bool = False,
    ) -> tuple:
        """
        对单张内容图进行风格迁移。

        Args:
            content:         内容 PIL RGB 图像
            style:           风格 PIL RGB 图像
            alpha:           风格强度 [0, 1]
            temporal_smooth: 是否使用帧间平滑（视频模式使用）

        Returns:
            (result_pil, inference_ms)
            result_pil:    PIL RGB 风格迁移结果
            inference_ms:  纯模型推理耗时（毫秒）
        """
        self._ensure_loaded()

        t_total_start = time.perf_counter()

        # 预处理：缩放
        content_resized = resize_keep_ratio(content, self.max_content_size)
        style_resized = resize_keep_ratio(style, self.max_style_size)

        orig_size = content_resized.size  # 用于后续还原（当前直接输出同尺寸）

        # PIL → Tensor
        content_t = pil_to_tensor(content_resized, self.device)
        style_t = pil_to_tensor(style_resized, self.device)

        # 推理
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            output_t = self._net(content_t, style_t, alpha=alpha)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t_infer_end = time.perf_counter()

        # Tensor → PIL
        result_pil = tensor_to_pil(output_t)

        # 还原到原始内容图尺寸
        if result_pil.size != content.size:
            result_pil = result_pil.resize(content.size, Image.LANCZOS)

        # 帧间平滑
        if temporal_smooth:
            result_pil = self._smoother.update(result_pil)

        t_total_end = time.perf_counter()

        self.last_inference_ms = (t_infer_end - t_infer_start) * 1000
        self.last_total_ms = (t_total_end - t_total_start) * 1000

        return result_pil, self.last_inference_ms

    def reset_smoother(self):
        """重置帧间平滑历史（切换风格时调用）。"""
        self._smoother.reset()

    def get_vram_mb(self) -> float:
        """
        返回当前 GPU 显存占用（MB）。
        CPU 模式下返回 0。
        """
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    # ─────────────────────────────────────────────────────────
    # 批量处理
    # ─────────────────────────────────────────────────────────

    def batch_transfer(
        self,
        input_dir: str,
        style_path: str,
        output_dir: str,
        alpha: float = 1.0,
        make_grid: bool = True,
        progress_callback=None,
    ) -> List[str]:
        """
        批量对文件夹内所有图像进行风格迁移。

        Args:
            input_dir:         内容图像文件夹
            style_path:        风格图像路径
            output_dir:        输出文件夹
            alpha:             风格强度
            make_grid:         是否输出对比网格图
            progress_callback: 进度回调 f(current, total, filename)

        Returns:
            output_paths: 所有输出文件路径列表
        """
        self._ensure_loaded()

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 收集图像文件
        img_files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in _IMG_EXTS
        ])
        if not img_files:
            raise ValueError(f"目录中没有找到图像文件：{input_dir}")

        style_pil = Image.open(style_path).convert("RGB")
        output_paths = []
        grid_pairs = []

        for i, img_file in enumerate(img_files):
            content_pil = Image.open(img_file).convert("RGB")
            result, ms = self.transfer(content_pil, style_pil, alpha=alpha)

            out_name = img_file.stem + "_styled" + img_file.suffix
            out_path = output_dir / out_name
            result.save(str(out_path))
            output_paths.append(str(out_path))

            if make_grid:
                grid_pairs.append((content_pil, result))

            if progress_callback:
                progress_callback(i + 1, len(img_files), img_file.name)

        # 生成对比网格图
        if make_grid and grid_pairs:
            grid_path = str(output_dir / "_comparison_grid.jpg")
            _make_comparison_grid(grid_pairs, grid_path)
            output_paths.append(grid_path)
            print(f"[StyleTransfer] 对比网格图已保存：{grid_path}")

        return output_paths

    # ─────────────────────────────────────────────────────────
    # 摄像头帧处理（供 GUI 调用）
    # ─────────────────────────────────────────────────────────

    def process_camera_frame(
        self,
        frame_bgr: np.ndarray,
        style_pil: Image.Image,
        alpha: float = 1.0,
    ) -> tuple:
        """
        处理摄像头的单帧（BGR numpy → 风格化 BGR numpy）。

        Args:
            frame_bgr:  OpenCV 读取的 BGR uint8 帧
            style_pil:  风格图像
            alpha:      风格强度

        Returns:
            (result_bgr, inference_ms)
        """
        content_pil = Image.fromarray(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result_pil, ms = self.transfer(
            content_pil, style_pil, alpha=alpha, temporal_smooth=True
        )
        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        return result_bgr, ms


# ─────────────────────────────────────────────────────────────
# 辅助：对比网格图
# ─────────────────────────────────────────────────────────────

def _make_comparison_grid(
    pairs: List[tuple],
    save_path: str,
    thumb_size: int = 256,
    cols: int = 4,
):
    """
    将 (content, result) 图像对拼接为网格对比图。

    Args:
        pairs:     [(content_pil, result_pil), ...]
        save_path: 保存路径
        thumb_size: 每张缩略图的边长
        cols:       每行显示的图像对数量（每对占2列）
    """
    from PIL import ImageDraw

    thumbs = []
    for content, result in pairs:
        c_t = content.copy()
        r_t = result.copy()
        c_t.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        r_t.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        # 加标注
        for img, label in ((c_t, "原图"), (r_t, "风格化")):
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, img.width, 16], fill=(0, 0, 0, 180))
            d.text((2, 1), label, fill=(255, 255, 255))
        thumbs.extend([c_t, r_t])

    n = len(thumbs)
    grid_cols = cols * 2
    grid_rows = (n + grid_cols - 1) // grid_cols

    grid = Image.new("RGB", (grid_cols * thumb_size,
                     grid_rows * thumb_size), (30, 30, 30))
    for idx, img in enumerate(thumbs):
        row = idx // grid_cols
        col = idx % grid_cols
        grid.paste(img, (col * thumb_size, row * thumb_size))

    grid.save(save_path, quality=90)


# ─────────────────────────────────────────────────────────────
# CLI 快速测试
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AdaIN 风格迁移 CLI")
    parser.add_argument("content", help="内容图像路径")
    parser.add_argument("style", help="风格图像路径")
    parser.add_argument("output", help="输出图像路径")
    parser.add_argument("--decoder", default=DEFAULT_DECODER_PATH,
                        help="decoder.pth 路径")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="风格强度 [0,1]，默认 1.0")
    args = parser.parse_args()

    st = StyleTransfer()
    st.load(args.decoder)

    content_pil = Image.open(args.content).convert("RGB")
    style_pil = Image.open(args.style).convert("RGB")

    result, ms = st.transfer(content_pil, style_pil, alpha=args.alpha)
    result.save(args.output)
    print(f"推理耗时：{ms:.1f} ms  → 已保存：{args.output}")
