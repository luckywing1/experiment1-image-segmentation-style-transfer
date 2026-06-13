"""
task_a/segmentation.py
DeepLabV3+ (ResNet-101) 语义分割模块
- 使用 torchvision 预训练权重，无需手动下载
- 支持 PASCAL VOC 21类语义分割
- 输出指定类别的二值掩码
"""

import ssl
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

# PASCAL VOC 21类标签
VOC_CLASSES = [
    ("背景",       "background"),
    ("飞机",       "aeroplane"),
    ("自行车",     "bicycle"),
    ("鸟",         "bird"),
    ("船",         "boat"),
    ("瓶子",       "bottle"),
    ("公共汽车",   "bus"),
    ("汽车",       "car"),
    ("猫",         "cat"),
    ("椅子",       "chair"),
    ("牛",         "cow"),
    ("餐桌",       "diningtable"),
    ("狗",         "dog"),
    ("马",         "horse"),
    ("摩托车",     "motorbike"),
    ("人",         "person"),
    ("盆栽",       "pottedplant"),
    ("羊",         "sheep"),
    ("沙发",       "sofa"),
    ("火车",       "train"),
    ("电视/显示器", "tvmonitor"),
]

# 中文名 -> 类别索引
CLASS_NAME_ZH_TO_IDX = {zh: i for i, (zh, _) in enumerate(VOC_CLASSES)}
CLASS_NAMES_ZH = [zh for zh, _ in VOC_CLASSES]

# 推理时的输入预处理
_INFERENCE_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class Segmentor:
    """
    DeepLabV3+ 分割器
    使用方法：
        seg = Segmentor()
        mask = seg.segment(pil_image, class_name_zh="人")
        # mask: np.ndarray uint8, shape (H, W), 前景=255, 背景=0
    """

    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        ssl._create_default_https_context = ssl._create_unverified_context

        print("[Segmentor] 正在加载 DeepLabV3+ 预训练权重（首次运行会自动下载，约 300 MB）...")
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.model = deeplabv3_resnet101(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        print(f"[Segmentor] 模型加载完成，运行设备：{self.device}")

    def segment(self, pil_image: Image.Image, class_name_zh: str = "人") -> np.ndarray:
        """
        对输入图像进行语义分割，返回指定类别的二值掩码。

        Args:
            pil_image: PIL RGB 图像
            class_name_zh: 目标类别中文名，如 "人"、"汽车" 等

        Returns:
            mask: np.ndarray uint8 (H, W)，前景=255，背景=0
        """
        self._ensure_loaded()

        if class_name_zh not in CLASS_NAME_ZH_TO_IDX:
            raise ValueError(f"未知类别：{class_name_zh}，可选类别：{CLASS_NAMES_ZH}")

        target_idx = CLASS_NAME_ZH_TO_IDX[class_name_zh]
        img_rgb = pil_image.convert("RGB")
        input_tensor = _INFERENCE_TRANSFORM(
            img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"]  # (1, 21, H, W)

        # argmax 得到每像素的预测类别
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # 生成二值掩码
        mask = np.zeros_like(pred, dtype=np.uint8)
        mask[pred == target_idx] = 255

        return mask

    def segment_all(self, pil_image: Image.Image) -> np.ndarray:
        """
        返回完整的类别预测图（每像素为类别索引 0~20）。
        用于可视化多类别分割结果。

        Returns:
            pred: np.ndarray uint8 (H, W)，值为 0~20
        """
        self._ensure_loaded()
        img_rgb = pil_image.convert("RGB")
        input_tensor = _INFERENCE_TRANSFORM(
            img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"]

        pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred

    def colorize_mask(self, pred_all: np.ndarray) -> np.ndarray:
        """
        将类别预测图映射为 RGB 伪彩色图，便于可视化。

        Args:
            pred_all: segment_all 的返回值

        Returns:
            color_mask: np.ndarray uint8 (H, W, 3)
        """
        # 使用固定调色板（PASCAL VOC 标准色彩）
        palette = _voc_palette()
        h, w = pred_all.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx in range(21):
            color_mask[pred_all == cls_idx] = palette[cls_idx]
        return color_mask

    def get_present_classes(self, pil_image: Image.Image) -> list:
        """
        返回图像中检测到的类别名称列表（排除背景）。
        可用于 GUI 动态更新可选类别。
        """
        pred_all = self.segment_all(pil_image)
        present = sorted(set(pred_all.flatten().tolist()))
        return [CLASS_NAMES_ZH[i] for i in present if i != 0]


def _voc_palette():
    """生成 PASCAL VOC 21 类伪彩色调色板"""
    palette = []
    for i in range(21):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette.append((r, g, b))
    return palette


if __name__ == "__main__":
    # 简单测试
    import sys
    if len(sys.argv) < 2:
        print("用法: python segmentation.py <图像路径> [类别中文名]")
        sys.exit(1)

    img_path = sys.argv[1]
    cls_name = sys.argv[2] if len(sys.argv) > 2 else "人"

    img = Image.open(img_path)
    seg = Segmentor()
    mask = seg.segment(img, cls_name)

    out_path = img_path.rsplit(".", 1)[0] + f"_mask_{cls_name}.png"
    Image.fromarray(mask).save(out_path)
    print(f"掩码已保存到：{out_path}")
