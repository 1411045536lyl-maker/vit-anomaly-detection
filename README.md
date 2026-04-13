# 基于 VIT 的工业级无监督异常检测

本项目使用 PyTorch 实现了一套基于 Vision Transformer (ViT) 的工业级无监督异常检测与缺陷定位系统。底层特征提取架构参考了纯 ViT 的实现思路，并通过引入 ImageNet 预训练先验知识，打破了传统目标检测强依赖海量缺陷标注数据的瓶颈。本系统仅需学习“正常态”样本，即可对未知表面缺陷进行像素级定位。

## 1. 项目结构

```text
.
|-- README.md
|-- train_memory_bank.py        # 训练脚本：构建正常样本特征记忆库
|-- evaluate.py                 # 评估脚本：计算全局 AUROC 并生成可视化热力图
|-- models
|   |-- __init__.py
|   `-- vit_extractor.py        # 核心架构：重构后的 ViT 空间特征提取器
|-- utils
|   |-- __init__.py
|   `-- dataset.py              # 数据模块：MVTec 数据集加载与预处理
|-- data
|   `-- mvtec_ad                # MVTec AD 数据集存放目录
|-- weights                     # 自动生成的正常样本特征记忆库
`-- outputs                     # 测试阶段生成的缺陷对比热力图
```

## 2. 已实现功能

- **预训练特征引擎**：引入官方 `vit_b_16` 预训练权重，极大提升对复杂工业纹理的泛化表征能力。
- **空间特征重构**：移除传统分类头，截取 Transformer 内部 Patch 序列并重构为二维空间特征图（Spatial Features）。
- **无监督记忆库构建**：在无梯度更新状态下提取正常样本特征，构建高维度、无重叠的特征记忆库（Memory Bank）。
- **零样本缺陷定位**：基于 KNN 思想与高效欧氏距离度量，计算测试特征的最短距离以生成异常得分。
- **高分辨率热力图**：通过双线性插值上采样，精准高亮细微划痕、形变与污染区域。
- **量化性能评估**：集成 Scikit-learn，自动计算并输出图像级 AUROC (Image-level AUROC) 指标。
- **命令行工程规范**：引入 `argparse` 解析器，消除硬编码，支持通过命令行一键切换不同工业产品类别的训练与评估。

## 3. 环境依赖

推荐使用 Python 3.9+，核心依赖如下：

```text
torch>=1.10.0
torchvision
opencv-python
scikit-learn
tqdm
Pillow
numpy
```

# 4. 数据集说明

本项目默认使用业界权威的无监督异常检测基准数据集 **MVTec AD** (MVTec Anomaly Detection)。  
该数据集包含 15 种不同的工业产品类别（5 种纹理，10 种物体），总计超过 5000 张高分辨率图像。

- **任务设定**: 训练集仅包含纯“正常”（Good）的工业品样本；测试集则包含正常样本以及带有各类真实缺陷（如划痕、裂纹、污染、形变等）的异常样本。

- **官方下载地址**: [MVTec AD Dataset Official Website](https://www.mvtectest.com/downloads)  

## 5. 快速开始

### 5.1 数据准备

请下载完整的 MVTec AD 数据集，并将其解压至项目根目录下的 `data/mvtec_ad/` 中，确保包含具体类别（如 `metal_nut`, `bottle` 等）及其内部的 `train/good` 与 `test` 等标准子目录。

### 5.2 构建特征记忆库

执行以下命令，提取训练集中所有正常样本的特征并持久化。初次运行将自动下载预训练权重。

```bash
# 默认处理 metal_nut 类别
python train_memory_bank.py

# 或者通过命令行参数指定其他类别（例如检测瓶子）
python train_memory_bank.py --category bottle
```

### 5.3 异常检测与性能评估

执行以下命令，模型将对测试集样本进行评分计算，在终端输出评估指标，并保存可视化热力图。

```bash
# 默认评估 metal_nut 类别
python evaluate.py

# 评估其他类别（需先构建对应的特征记忆库）
python evaluate.py --category bottle
```

## 6. 输出结果

在 MVTec AD (Metal Nut) 测试集中，本系统实现了 **99.95% 的 Image-level AUROC**。

训练与评估完成后，项目中将自动生成以下文件结构：

- `weights/{category}/memory_bank.pt`: 固化的正常态特征记忆库张量。
- `outputs/{category}/heatmaps/`: 包含所有测试样本的对比结果图。图像左侧为还原后的输入原图，右侧为叠加了伪彩色（COLORMAP_JET）的缺陷高亮热力图。
