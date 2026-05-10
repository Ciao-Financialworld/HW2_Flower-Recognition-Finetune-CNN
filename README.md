# Task 1：Flowers-102 图像分类课程作业

本项目是《深度学习与空间智能》期中作业 `task1` 的代码提交版本。实验基于 `torchvision` 提供的 `Flowers102` 数据集，主要完成预训练模型微调、注意力模块对比、超参数分析和预训练消融实验。



## 提交信息

- Github repo: `https://github.com/Ciao-Financialworld/HW2_Flower-Recognition-Finetune-CNN.git`
- 实验报告提交形式：PDF
- 本仓库用途：提供代码、环境配置、训练方法、测试方法和模型权重说明

## 项目内容

当前仓库包含以下几类实验：

1. `ResNet-18` 基线模型  
   使用 ImageNet 预训练权重进行微调。

2. 注意力模块对比  
   在 `ResNet-18` 基础上实现了：
   - `SE-block`
   - `CBAM`

3. Transformer 模型对比  
   使用 `ViT-Tiny` 进行分类实验。

4. 超参数分析  
   以预训练 `ResNet-18` 为基础，对以下因素进行对比：
   - 学习率组合
   - batch size
   - 训练 epoch 数
   - 学习率调度策略

5. 预训练消融实验  
   对比 `pretrained=True` 和 `pretrained=False` 两种初始化方式。

## 目录结构

```text
task1/
├── utils.py                        # 数据加载、训练与评估公共函数
├── train_resnet18_baseline.py      # ResNet-18 基线模型
├── train_resnet18_se.py            # ResNet-18 + SE-block
├── train_resnet18_cbam.py          # ResNet-18 + CBAM
├── train_vit_tiny.py               # ViT-Tiny
├── train_hyperparam_analysis.py    # 超参数分析
├── ablation_pretrain.py            # 预训练消融实验
├── test.py                         # 测试脚本
├── export_notebook_json.py         # notebook 导出辅助脚本
├── export_notebook_json.ipynb
├── checkpoints/                    # 保存的模型权重
├── data/                           # Flowers-102 数据
└── README.md
```

## 环境配置

建议使用 Python 3.10 及以上版本。

```bash
pip install torch torchvision
pip install swanlab
pip install timm
```

说明：

- `torch`、`torchvision` 是训练和数据集加载所必需的。
- `swanlab` 用于保存本地训练日志。
- `timm` 仅在 `ViT-Tiny` 和 `Swin-Tiny` 相关模型测试时需要。

## 数据集说明

项目使用 `Flowers102` 数据集，默认通过 `torchvision.datasets.Flowers102` 下载或读取，数据目录为 `./data`。

官方划分如下：

- train: 1020
- val: 1020
- test: 6149
- classes: 102

如果本地已经存在数据，脚本会直接复用，不需要重复准备。

## 详细实验设置

以下配置对应当前仓库中的主实验脚本 `train_resnet18_baseline.py`、`train_resnet18_se.py`、`train_resnet18_cbam.py` 和 `train_vit_tiny.py`。

| 项目 | 设置 |
| --- | --- |
| 数据集 | Flowers102 |
| 数据划分 | train=1020, val=1020, test=6149 |
| 输入尺寸 | 224 x 224 |
| 主干模型 | ResNet-18 / ResNet-18+SE / ResNet-18+CBAM / ViT-Tiny |
| 预训练 | 默认使用 ImageNet 预训练权重 |
| batch size | 32 |
| epoch | 40 |
| optimizer | AdamW |
| learning rate | `lr_head=1e-3`, `lr_backbone=1e-4` |
| weight decay | `1e-2` |
| scheduler | warmup + cosine decay |
| warmup epochs | 5 |
| loss function | CrossEntropyLoss with `label_smoothing=0.1` |
| 评价指标 | Top-1 Accuracy, Top-5 Accuracy |
| 训练设备 | 自动检测 `cuda`，否则使用 `cpu` |

补充说明：

- 对于基线和注意力模型，分类头使用较大学习率，backbone 使用较小学习率进行微调。
- 训练集每个 epoch 的 iteration 数与 batch size 有关。以 `batch_size=32` 为例，train 集 1020 张图像约为 32 个 iteration/epoch。
- 超参数分析脚本会额外测试不同的学习率组合、batch size、epoch 数和调度策略。

## 训练方法

### 1. 基线模型

```bash
python train_resnet18_baseline.py
```

### 2. 注意力模型

```bash
python train_resnet18_se.py
python train_resnet18_cbam.py
```

### 3. ViT-Tiny

```bash
python train_vit_tiny.py
```

### 4. 超参数分析

```bash
python train_hyperparam_analysis.py --analysis lr
python train_hyperparam_analysis.py --analysis batch
python train_hyperparam_analysis.py --analysis epoch
python train_hyperparam_analysis.py --analysis scheduler
python train_hyperparam_analysis.py --analysis all
```

超参数分析范围如下：

- 学习率组合：`lr_head` 和 `lr_backbone`
- batch size：16 / 32 / 64 / 128
- epoch：20 / 40 / 60 / 80
- scheduler：`cosine` / `step` / `multistep` / `constant`

### 5. 预训练消融

```bash
python ablation_pretrain.py
```

## 测试方法

测试单个模型：

```bash
python test.py --model resnet18_baseline --ckpt ./checkpoints/resnet18_baseline_best.pth
python test.py --model resnet18_se --ckpt ./checkpoints/resnet18_se_best.pth
python test.py --model resnet18_cbam --ckpt ./checkpoints/resnet18_cbam_best.pth
python test.py --model vit_tiny --ckpt ./checkpoints/vit_tiny_best.pth
```

测试 `checkpoints/` 下所有已保存模型：

```bash
python test.py --all
```

测试脚本会输出：

- Test Loss
- Top-1 Accuracy
- Top-5 Accuracy

## 输出文件

训练完成后，主要输出包括：

- `./checkpoints/`：各模型最佳权重
- `./checkpoints/hpsearch/`：超参数分析过程中保存的权重
- `./swanlab_logs/`：本地训练日志

## 可视化记录

课程要求中提到需要给出训练过程可视化截图。本项目使用 `swanlab` 本地模式记录训练日志。

启动方式：

```bash
swanlab watch ./swanlab_logs
```

建议在实验报告中截图以下内容：

- 训练集和验证集的 `loss` 曲线
- 验证集 `accuracy` 曲线
- 不同模型或不同超参数设置的对比曲线

说明：

- 本任务是图像分类任务，因此主要使用 `Accuracy`，而不是目标检测中的 `mAP`。
- 当前训练脚本会记录 `train/loss`、`train/accuracy`、`val/loss`、`val/accuracy`、`test/accuracy` 和学习率变化。

## 模型权重下载

课程要求中需要提供模型权重的网盘下载地址。本项目训练得到的模型权重统一托管在 Google Drive：

`https://drive.google.com/drive/folders/1wfNy5JJI22H8WwRlFdjM4Pbi3XZsOY6?usp=drive_link`

| 模型 | 权重文件 | 下载地址 |
| --- | --- | --- |
| ResNet-18 Baseline | `checkpoints/resnet18_baseline_best.pth` | Google Drive 文件夹链接 |
| ResNet-18 + SE | `checkpoints/resnet18_se_best.pth` | Google Drive 文件夹链接 |
| ResNet-18 + CBAM | `checkpoints/resnet18_cbam_best.pth` | Google Drive 文件夹链接 |
| ViT-Tiny | `checkpoints/vit_tiny_best.pth` | Google Drive 文件夹链接 |
| 预训练消融（Scratch） | `checkpoints/resnet18_scratch_best.pth` | Google Drive 文件夹链接 |



## 备注

如果需要查看本地日志，可以使用：

```bash
swanlab watch ./swanlab_logs
```


