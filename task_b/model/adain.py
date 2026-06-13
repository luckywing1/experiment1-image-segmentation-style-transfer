"""
task_b/model/adain.py
AdaIN 风格迁移网络定义
参考论文：Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
          Huang & Belongie, ICCV 2017

网络结构：
  编码器：VGG19 前4个 block（到 relu4_1），使用 vgg_normalised.pth 权重
  AdaIN 层：调整内容特征统计量以匹配风格特征
  解码器：与编码器对称的上采样网络（需加载预训练权重）
"""

import os
import ssl
import urllib.request

import torch
import torch.nn as nn

ssl._create_default_https_context = ssl._create_unverified_context

# vgg_normalised.pth 下载地址与默认保存路径
_VGG_NORMALISED_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/"
    "releases/latest/download/vgg_normalised.pth"
)
DEFAULT_VGG_NORMALISED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vgg_normalised.pth"
)

# decoder.pth 下载地址与默认保存路径
_DECODER_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/"
    "releases/latest/download/decoder.pth"
)
DEFAULT_DECODER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "decoder.pth"
)


def download_vgg_normalised(
    save_path: str = DEFAULT_VGG_NORMALISED_PATH,
    progress_callback=None,
) -> str:
    """下载 vgg_normalised.pth 。"""
    return _download_weight(
        url=_VGG_NORMALISED_URL,
        save_path=save_path,
        name="vgg_normalised.pth",
        progress_callback=progress_callback,
    )


def download_decoder(
    save_path: str = DEFAULT_DECODER_PATH,
    progress_callback=None,
) -> str:
    """下载 decoder.pth。"""
    return _download_weight(
        url=_DECODER_URL,
        save_path=save_path,
        name="decoder.pth",
        progress_callback=progress_callback,
    )


def _download_weight(
    url: str,
    save_path: str,
    name: str,
    progress_callback=None,
) -> str:
    """通用权重文件下载函数。"""
    if os.path.exists(save_path):
        return save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[AdaIN] Downloading {name} -> {save_path}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        print(f"\r  {pct:.0f}%  ({downloaded/1048576:.1f}/{total_size/1048576:.1f} MB)",
              end="", flush=True)
        if progress_callback:
            progress_callback(pct, downloaded, total_size)

    try:
        urllib.request.urlretrieve(url, save_path, _progress)
        print(f"\n[AdaIN] Download complete: {save_path}")
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise RuntimeError(
            f"Failed to download {name}: {e}\n"
            f"Please download manually:\n  {url}"
        ) from e

    return save_path


# ─────────────────────────────────────────────────────────────
# AdaIN 核心操作
# ─────────────────────────────────────────────────────────────

def adaptive_instance_normalization(content_feat: torch.Tensor,
                                    style_feat: torch.Tensor) -> torch.Tensor:
    """AdaIN: 将内容特征的统计量替换为风格特征的统计量。"""
    size = content_feat.size()
    style_mean = style_feat.view(
        size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1)
    style_std = style_feat.view(
        size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1) + 1e-6
    content_mean = content_feat.view(
        size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1)
    content_std = content_feat.view(
        size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1) + 1e-6
    return (content_feat - content_mean) / content_std * style_std + style_mean


# ─────────────────────────────────────────────────────────────
# 构建VGG Sequential
# ─────────────────────────────────────────────────────────────

def _build_vgg_normalised_sequential() -> nn.Sequential:
    """
    构建与 naoto0804/pytorch-AdaIN 的 vgg_normalised.pth 完全匹配的 Sequential。

    结构（52 层，indices 0-51）：
      Conv2d(3,3,1)        # 0  归一化层 (1x1 conv, mean subtraction)
      ReflectionPad2d(1)   # 1
      Conv2d(3,64,3,pad=0) # 2  conv1_1
      ReLU                 # 3
      ReflectionPad2d(1)   # 4
      Conv2d(64,64,3,0)    # 5  conv1_2
      ReLU                 # 6
      AvgPool2d(2,2)       # 7  pool1
      ReflectionPad2d(1)   # 8
      Conv2d(64,128,3,0)   # 9  conv2_1
      ReLU                 # 10
      ReflectionPad2d(1)   # 11
      Conv2d(128,128,3,0)  # 12 conv2_2
      ReLU                 # 13
      AvgPool2d(2,2)       # 14 pool2
      ReflectionPad2d(1)   # 15
      Conv2d(128,256,3,0)  # 16 conv3_1
      ReLU                 # 17
      ReflectionPad2d(1)   # 18
      Conv2d(256,256,3,0)  # 19 conv3_2
      ReLU                 # 20
      ReflectionPad2d(1)   # 21
      Conv2d(256,256,3,0)  # 22 conv3_3
      ReLU                 # 23
      ReflectionPad2d(1)   # 24
      Conv2d(256,256,3,0)  # 25 conv3_4
      ReLU                 # 26
      AvgPool2d(2,2)       # 27 pool3
      ReflectionPad2d(1)   # 28
      Conv2d(256,512,3,0)  # 29 conv4_1
      ReLU                 # 30
      ReflectionPad2d(1)   # 31
      Conv2d(512,512,3,0)  # 32 conv4_2
      ReLU                 # 33
      ReflectionPad2d(1)   # 34
      Conv2d(512,512,3,0)  # 35 conv4_3
      ReLU                 # 36
      ReflectionPad2d(1)   # 37
      Conv2d(512,512,3,0)  # 38 conv4_4
      ReLU                 # 39
      AvgPool2d(2,2)       # 40 pool4
      ReflectionPad2d(1)   # 41
      Conv2d(512,512,3,0)  # 42 conv5_1
      ReLU                 # 43
      ReflectionPad2d(1)   # 44
      Conv2d(512,512,3,0)  # 45 conv5_2
      ReLU                 # 46
      ReflectionPad2d(1)   # 47
      Conv2d(512,512,3,0)  # 48 conv5_3
      ReLU                 # 49
      ReflectionPad2d(1)   # 50
      Conv2d(512,512,3,0)  # 51 conv5_4
    """
    return nn.Sequential(
        # 归一化层
        nn.Conv2d(3, 3, 1),                                  # 0
        # Block 1
        nn.ReflectionPad2d(1), nn.Conv2d(3, 64, 3),          # 1, 2
        nn.ReLU(inplace=False),                               # 3
        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3),         # 4, 5
        nn.ReLU(inplace=False),                               # 6
        nn.AvgPool2d(2, 2),                                   # 7
        # Block 2
        nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3),        # 8, 9
        nn.ReLU(inplace=False),                               # 10
        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3),       # 11, 12
        nn.ReLU(inplace=False),                               # 13
        nn.AvgPool2d(2, 2),                                   # 14
        # Block 3
        nn.ReflectionPad2d(1), nn.Conv2d(128, 256, 3),       # 15, 16
        nn.ReLU(inplace=False),                               # 17
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),       # 18, 19
        nn.ReLU(inplace=False),                               # 20
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),       # 21, 22
        nn.ReLU(inplace=False),                               # 23
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),       # 24, 25
        nn.ReLU(inplace=False),                               # 26
        nn.AvgPool2d(2, 2),                                   # 27
        # Block 4
        nn.ReflectionPad2d(1), nn.Conv2d(256, 512, 3),       # 28, 29
        nn.ReLU(inplace=False),                               # 30
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),       # 31, 32
        nn.ReLU(inplace=False),                               # 33
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),       # 34, 35
        nn.ReLU(inplace=False),                               # 36
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),       # 37, 38
        nn.ReLU(inplace=False),                               # 39
        nn.AvgPool2d(2, 2),                                   # 40
        # Block 5
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),       # 41, 42
        nn.ReLU(inplace=False),                               # 43
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),       # 44, 45
        nn.ReLU(inplace=False),                               # 46
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),       # 47, 48
        nn.ReLU(inplace=False),                               # 49
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),       # 50, 51
    )


# ─────────────────────────────────────────────────────────────
# VGG19 编码器
# ─────────────────────────────────────────────────────────────

class VGGEncoder(nn.Module):
    """
    VGG19 编码器，与 naoto0804/pytorch-AdaIN 原始实现完全一致。
    使用 vgg_normalised.pth 权重（含归一化层 + ReflectionPad + AvgPool）。
    编码器输出为 relu4_1（index 30），与 AdaIN 论文一致。
    输入仅需 ToTensor 到 [0, 1]，不需要 VGG 归一化。
    """

    def __init__(self, vgg_normalised_path: str = None):
        super().__init__()

        loaded = False

        # 尝试加载 vgg_normalised.pth
        for path in [vgg_normalised_path, DEFAULT_VGG_NORMALISED_PATH]:
            if path and os.path.exists(path):
                try:
                    self._load_from_vgg_normalised(path)
                    loaded = True
                    break
                except Exception as e:
                    print(f"[AdaIN] Failed to load {path}: {e}")

        if not loaded:
            try:
                path = download_vgg_normalised()
                self._load_from_vgg_normalised(path)
                loaded = True
            except Exception:
                pass

        if not loaded:
            print(
                "[AdaIN] WARNING: vgg_normalised.pth unavailable, using torchvision fallback")
            print(f"       Download from: {_VGG_NORMALISED_URL}")
            self._build_from_torchvision()

        for param in self.parameters():
            param.requires_grad = False

    def _load_from_vgg_normalised(self, path: str):
        """从 vgg_normalised.pth 加载编码器。"""
        data = torch.load(path, map_location="cpu")

        if isinstance(data, dict):
            vgg = _build_vgg_normalised_sequential()
            vgg.load_state_dict(data)
        elif isinstance(data, nn.Module):
            vgg = data
        else:
            raise ValueError(f"Unexpected type: {type(data)}")

        # 编码器只需 layers 0-30（norm → relu4_1）
        # 与 AdaIN 论文一致：内容特征取自 relu4_1 (block4_conv1)
        # 解码器 9 层 Conv2d + 3 层 Upsample 正好对应从 relu4_1 反向重构
        all_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*all_layers[0:4])    # norm→relu1_1
        # pad→conv1_2→relu1_2→pool1→pad→conv2_1→relu2_1
        self.enc_2 = nn.Sequential(*all_layers[4:11])
        # pad→conv2_2→relu2_2→pool2→pad→conv3_1→relu3_1
        self.enc_3 = nn.Sequential(*all_layers[11:18])
        # pad→conv3_2→...→pool3→pad→conv4_1→relu4_1
        self.enc_4 = nn.Sequential(*all_layers[18:31])

        print("[AdaIN] VGG encoder loaded from vgg_normalised.pth")

    def _build_from_torchvision(self):
        """回退方案：从 torchvision VGG19 构建。"""
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = []
        for layer in vgg.features:
            if isinstance(layer, nn.MaxPool2d):
                features.append(nn.AvgPool2d(2, 2))
            elif isinstance(layer, nn.ReLU):
                features.append(nn.ReLU(inplace=False))
            else:
                features.append(layer)
        seq = nn.Sequential(*features)
        children = list(seq.children())
        # relu4_1 = index 20 in standard VGG19 features
        self.enc_1 = nn.Sequential(*children[0:2])    # conv1_1, relu1_1
        # conv1_2, relu1_2, pool1, conv2_1, relu2_1
        self.enc_2 = nn.Sequential(*children[2:7])
        # conv2_2, relu2_2, pool2, conv3_1, relu3_1
        self.enc_3 = nn.Sequential(*children[7:12])
        # conv3_2,...,pool3, conv4_1, relu4_1
        self.enc_4 = nn.Sequential(*children[12:21])

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x:          (B, 3, H, W) [0,1] 范围输入
            return_all: 是否返回中间层特征（训练用）
        """
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        if return_all:
            return [h1, h2, h3, h4]
        return h4


# ─────────────────────────────────────────────────────────────
# 解码器
# ─────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """AdaIN 解码器，需加载 decoder.pth 预训练权重。"""

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
    """AdaIN 风格迁移完整网络。"""

    def __init__(self, vgg_normalised_path: str = None):
        super().__init__()
        self.encoder = VGGEncoder(vgg_normalised_path)
        self.decoder = Decoder()

    def load_decoder(self, weight_path: str):
        """加载解码器预训练权重（自动处理键名前缀）。"""
        state = torch.load(weight_path, map_location="cpu")
        if not any(k.startswith("net.") for k in state.keys()):
            state = {"net." + k: v for k, v in state.items()}
        self.decoder.load_state_dict(state)
        print(f"[AdaINNet] Decoder weights loaded: {weight_path}")

    def encode(self, x: torch.Tensor):
        return self.encoder(x, return_all=False)

    def encode_all(self, x: torch.Tensor):
        return self.encoder(x, return_all=True)

    def forward(self, content: torch.Tensor, style: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        """
        Args:
            content: (1, 3, H, W) [0,1] 内容图
            style:   (1, 3, H', W') [0,1] 风格图
            alpha:   风格强度 [0, 1]
        Returns:
            output: (1, 3, H, W) [0,1] 风格迁移结果
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        content_feat = self.encoder(content, return_all=False)
        style_feat = self.encoder(style, return_all=False)
        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1.0 - alpha) * content_feat
        return self.decoder(t)
