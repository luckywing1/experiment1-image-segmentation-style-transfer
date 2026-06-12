"""
task_b/model/adain.py
AdaIN 风格迁移网络定义
参考论文：Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
          Huang & Belongie, ICCV 2017

网络结构：
  编码器：VGG19 前4个 block（到 relu4_1），权重从 torchvision 加载
  AdaIN 层：调整内容特征统计量以匹配风格特征
  解码器：与编码器对称的上采样网络（需加载预训练权重）
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────────────────────
# AdaIN 核心操作
# ─────────────────────────────────────────────────────────────

def adaptive_instance_normalization(content_feat: torch.Tensor,
                                    style_feat: torch.Tensor) -> torch.Tensor:
    """
    AdaIN 操作：将 content_feat 的统计量替换为 style_feat 的统计量。

    Args:
        content_feat: (B, C, H, W) 内容特征
        style_feat:   (B, C, H', W') 风格特征

    Returns:
        normalized: (B, C, H, W) 经 AdaIN 归一化的特征
    """
    assert content_feat.size()[:2] == style_feat.size()[:2], \
        "内容与风格特征的 batch/channel 维度必须一致"

    size = content_feat.size()

    # 计算统计量（沿 H, W 维度）
    style_mean = style_feat.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)
    style_std = style_feat.view(size[0], size[1], -1).std(dim=2).view(size[0], size[1], 1, 1) + 1e-6

    content_mean = content_feat.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)
    content_std = content_feat.view(size[0], size[1], -1).std(dim=2).view(size[0], size[1], 1, 1) + 1e-6

    # 归一化内容特征，再用风格统计量缩放
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


# ─────────────────────────────────────────────────────────────
# VGG19 编码器（截取到 relu4_1）
# ─────────────────────────────────────────────────────────────

class VGGEncoder(nn.Module):
    """
    VGG19 编码器，提取 relu1_1 / relu2_1 / relu3_1 / relu4_1 特征。
    权重从 torchvision 预训练 VGG19 加载，冻结参数。
    """

    # VGG19 各层到 relu4_1 的层索引边界
    _RELU_LAYERS = {
        "relu1_1": 2,
        "relu2_1": 7,
        "relu3_1": 12,
        "relu4_1": 21,
    }

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = list(vgg.features.children())

        # 切成 4 个子网段
        self.slice1 = nn.Sequential(*features[0:2])    # → relu1_1
        self.slice2 = nn.Sequential(*features[2:7])    # → relu2_1
        self.slice3 = nn.Sequential(*features[7:12])   # → relu3_1
        self.slice4 = nn.Sequential(*features[12:21])  # → relu4_1

        # 冻结编码器参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x:          (B, 3, H, W) 输入图像张量（已 VGG 归一化）
            return_all: True → 返回所有中间层特征列表，用于损失计算
                        False → 仅返回 relu4_1 特征

        Returns:
            若 return_all=False：relu4_1 特征张量
            若 return_all=True：[relu1_1, relu2_1, relu3_1, relu4_1]
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if return_all:
            return [h1, h2, h3, h4]
        return h4


# ─────────────────────────────────────────────────────────────
# 解码器（需加载预训练权重）
# ─────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    AdaIN 解码器，与 VGG19 编码器（到 relu4_1）对称。
    需加载 naoto0804/pytorch-AdaIN 提供的预训练权重：
        decoder.pth  （约 32 MB）
    下载地址见 README.md
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 4 反向
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Block 3 反向
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Block 2 反向
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Block 1 反向
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# AdaIN 完整网络
# ─────────────────────────────────────────────────────────────

class AdaINNet(nn.Module):
    """
    AdaIN 风格迁移完整网络。
    使用方法：
        net = AdaINNet()
        net.load_decoder("path/to/decoder.pth")
        net.to(device).eval()

        output = net(content_tensor, style_tensor, alpha=0.8)
    """

    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()

    def load_decoder(self, weight_path: str):
        """加载解码器预训练权重。"""
        state = torch.load(weight_path, map_location="cpu")
        self.decoder.load_state_dict(state)
        print(f"[AdaINNet] 解码器权重已加载：{weight_path}")

    def encode(self, x: torch.Tensor):
        """仅编码，返回 relu4_1 特征。"""
        return self.encoder(x, return_all=False)

    def encode_all(self, x: torch.Tensor):
        """编码并返回所有中间层特征（用于损失计算）。"""
        return self.encoder(x, return_all=True)

    def forward(self, content: torch.Tensor, style: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        """
        Args:
            content: (1, 3, H, W) 内容图像张量
            style:   (1, 3, H', W') 风格图像张量
            alpha:   风格化强度 [0, 1]；0=完全保留内容，1=完全风格化

        Returns:
            output: (1, 3, H, W) 风格迁移结果张量
        """
        alpha = float(np.clip(alpha, 0.0, 1.0)) if _np_available else max(0.0, min(1.0, float(alpha)))

        # 提取特征
        content_feat = self.encoder(content, return_all=False)
        style_feat = self.encoder(style, return_all=False)

        # AdaIN
        t = adaptive_instance_normalization(content_feat, style_feat)

        # alpha 插值：在特征空间混合
        t = alpha * t + (1.0 - alpha) * content_feat

        # 解码
        output = self.decoder(t)
        return output


# 可选 numpy clip
try:
    import numpy as np
    _np_available = True
except ImportError:
    _np_available = False
