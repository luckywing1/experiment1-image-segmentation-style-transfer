# 智能计算系统结课作业 · 实验一

## 项目简介

本项目实现实验一的两个任务：

- **任务A**：基于 DeepLabV3+（ResNet-101）的语义分割，提取图像前景，支持 Alpha 融合与泊松融合将前景合成到新背景。
- **任务B**：基于 AdaIN 的实时图像风格迁移，支持风格强度调节、摄像头实时迁移、批量处理。

---

## 目录结构

```
结课作业/
├── task_a/
│   ├── segmentation.py       # DeepLabV3+ 分割模块
│   ├── compositing.py        # 融合/羽化处理
│   └── gui_task_a.py         # 任务A 桌面GUI（入口）
├── task_b/
│   ├── model/
│   │   ├── adain.py          # AdaIN 网络结构
│   │   └── transform.py      # 图像预处理/后处理
│   ├── style_transfer.py     # 风格迁移推理引擎
│   ├── decoder.pth           # ← 需要手动下载（见下方说明）
│   └── gui_task_b.py         # 任务B 桌面GUI（入口）
├── assets/
│   ├── style_images/         # 放入风格参考图（任意 jpg/png）
│   └── test_images/          # 放入测试内容图
├── requirements.txt
└── README.md
```

---

## 环境要求

| 项目     | 要求                    |
| -------- | ----------------------- |
| Python   | 3.9                     |
| CUDA     | 可选                    |
| 操作系统 | Windows / Linux / macOS |

---

## 一、安装环境

### 1. 创建 conda 虚拟环境

```bash
conda create -n cv_exp python=3.9 -y
conda activate cv_exp
```

### 2. 安装 PyTorch

你的显卡为 **RTX 4060 Laptop**，驱动 CUDA 版本 13.2，直接使用以下命令（PyTorch 官方最新稳定版支持到 CUDA 12.x，向下兼容，推荐 cu124）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> 如遇网络问题可尝试先从 [https://pytorch.org](https://pytorch.org) 官网获取完整包名再手动安装。

**CPU 版本（不推荐，速度较慢）：**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

---

## 二、你需要额外准备的内容

### 任务A（图像分割）

- **原图**：任意一张含有清晰前景物体的 jpg/png 图片（如含有人、猫、狗、车的照片）
- **背景图（可选）**：任意一张你想换上的背景图片；不提供则自动使用白色背景

> DeepLabV3+ 预训练权重会在**首次运行时自动下载**（约 300 MB），需要网络连接。

### 任务B（风格迁移）

- **内容图**：任意一张你想进行风格化的 jpg/png 图片
- **风格参考图**：任意一张画作或风格图（如梵高星夜、莫奈睡莲等）；也可放入 `assets/style_images/` 目录作为内置预设
- **decoder.pth（必须）**：AdaIN 解码器预训练权重文件（约 32 MB）

  **下载方式（二选一）：**

  方式1 — 浏览器下载：

  ```
  https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.1.0/decoder.pth
  ```

  方式2 — 命令行下载：

  ```bash
  # Windows（需要 curl）
  curl -L -o task_b/decoder.pth https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.1.0/decoder.pth
  ```

  下载后将 `decoder.pth` 放入 `task_b/` 目录（程序会自动检测）。

---

## 三、运行方式

### 运行任务A（图像分割与前景合成）

```bash
conda activate cv_exp
python task_a/gui_task_a.py
```

**操作步骤：**

1. 点击「加载原图」选择含前景物体的图片
2. 在「目标类别」下拉框选择要提取的类别（如「人」「汽车」「猫」等）
3. 点击「加载背景图」（可选）
4. 调节「羽化强度」滑块，值越大边缘越柔和
5. 选择「Alpha 融合」或「泊松融合」
6. 点击「▶ 运行分割与合成」
7. 查看四联预览（原图 / 掩码 / 提取前景 / 合成结果），点击「保存」保存结果

### 运行任务B（实时图像风格迁移）

```bash
conda activate cv_exp
python task_b/gui_task_b.py
```

**操作步骤：**

1. 点击「加载内容图」选择要进行风格化的图片
2. 点击「加载风格图」选择风格参考图，或从「内置风格」下拉框选择
3. 若 `decoder.pth` 未自动找到，点击「选择权重文件」手动指定
4. 调节「风格强度」滑块（0=保留原图，100=完全风格化）
5. 点击「▶ 风格迁移」进行单张图片迁移
6. 点击「开启摄像头」进行实时视频流风格迁移（需要摄像头）
7. 点击「批量处理文件夹」对整个文件夹的图片批量风格化
8. 底部状态栏显示推理耗时（ms）和 GPU 显存占用

---

## 四、内置风格图说明

将任意 jpg/png 风格参考图放入 `assets/style_images/` 目录后，重启程序，
即可在「内置风格」下拉框中看到这些图片，方便快速切换。

---

## 五、推荐测试图片来源

| 用途          | 推荐来源                                             |
| ------------- | ---------------------------------------------------- |
| 内容图 / 原图 | 手机拍摄照片，Unsplash 免费图                        |
| 风格参考图    | WikiArt（wikimedia 上的名画）、Google 搜索印象派画作 |
| 背景图        | Unsplash、Pexels 免费背景图                          |

---

## 六、常见问题

**Q：首次运行任务A时等待很久？**
A：DeepLabV3+ 权重约 300 MB，首次自动下载需要时间，后续运行无需重复下载。

**Q：任务B提示"decoder.pth 不存在"？**
A：按上方说明下载 decoder.pth 并放入 `task_b/` 目录，或在 GUI 中点击「选择权重文件」。

**Q：CPU 运行任务B很慢？**
A：在「内容图最大边」下拉框选择 256 或 384 可以显著提升速度（牺牲部分分辨率）。

**Q：泊松融合结果有色差？**
A：泊松融合适合光照接近的场景，光差大时建议使用 Alpha 融合。

**Q：摄像头模式画面延迟大？**
A：建议在 GPU 上运行，或将「内容图最大边」设置为 256。
