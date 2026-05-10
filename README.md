# Task 1：微调预训练 CNN 实现花卉识别

基于 **102 Category Flower Dataset** 完成以下四个子任务：
1. ResNet-18 基准模型（ImageNet 预训练微调）
2. 超参数分析（学习率 / batch size / epoch / 调度策略）
3. 预训练消融实验（pretrained vs scratch）
4. 注意力机制对比（SE-block / CBAM / ViT-Tiny / Swin-T）

---

## 环境配置

**Python ≥ 3.10，CUDA ≥ 11.8（推荐）**

```bash
# 1. 创建虚拟环境（推荐）
conda create -n flower102 python=3.10 -y
conda activate flower102

# 2. 安装 PyTorch（根据 CUDA 版本选择，见 https://pytorch.org）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install timm        # ViT-Tiny、Swin-T
pip install swanlab     # 训练可视化（本地模式，无需账号）
```

**验证安装：**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import timm; print(timm.__version__)"
python -c "import swanlab; print(swanlab.__version__)"
```

如果预训练权重下载较慢，可在当前终端先设置 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 文件结构

```
task1/
├── utils.py                        # 数据加载、训练/评估公共函数
├── train_resnet18_baseline.py      # 子任务①：ResNet-18 基准
├── train_resnet18_se.py            # 子任务④：ResNet-18 + SE-block
├── train_resnet18_cbam.py          # 子任务④：ResNet-18 + CBAM
├── train_vit_tiny.py               # 子任务④：ViT-Tiny
├── train_swin_t.py                 # 子任务④：Swin-Tiny
├── train_hyperparam_analysis.py    # 子任务②：超参数系统分析
├── ablation_pretrain.py            # 子任务③：预训练消融实验
├── test.py                         # 测试脚本（评估已保存的模型）
└── README.md
```

---

## 数据集

102 Category Flower Dataset 首次运行时**自动下载**（通过 torchvision），无需手动准备。

| 划分 | 样本数 | 类别数 |
|------|--------|--------|
| train | 1,020 | 102 |
| val   | 1,020 | 102 |
| test  | 6,149 | 102 |

---

## 训练

### 子任务① 基准模型

```bash
python train_resnet18_baseline.py
```

### 子任务④ 注意力机制变体

```bash
python train_resnet18_se.py      # SE-block
python train_resnet18_cbam.py    # CBAM
python train_vit_tiny.py         # ViT-Tiny
python train_swin_t.py           # Swin-Tiny
```

### 子任务② 超参数分析

```bash
# 分析所有维度（耗时较长）
python train_hyperparam_analysis.py --analysis all

# 只分析某一维度
python train_hyperparam_analysis.py --analysis lr         # 学习率组合
python train_hyperparam_analysis.py --analysis batch      # Batch Size
python train_hyperparam_analysis.py --analysis epoch      # 训练 Epoch 数
python train_hyperparam_analysis.py --analysis scheduler  # 调度策略
```

### 子任务③ 预训练消融实验

```bash
python ablation_pretrain.py
# 自动依次运行：ImageNet预训练微调 vs 随机初始化从零训练
# 训练结束后打印两组 Test Acc 对比及提升幅度
```

---

## 测试 / 评估

训练完成后，checkpoint 自动保存至 `./checkpoints/<model_name>_best.pth`。

```bash
# 评估单个模型（Top-1 + Top-5 准确率）
python test.py --model resnet18_baseline \
               --ckpt ./checkpoints/resnet18_baseline_best.pth

python test.py --model resnet18_se \
               --ckpt ./checkpoints/resnet18_se_best.pth

python test.py --model resnet18_cbam \
               --ckpt ./checkpoints/resnet18_cbam_best.pth

python test.py --model vit_tiny \
               --ckpt ./checkpoints/vit_tiny_best.pth

python test.py --model swin_tiny \
               --ckpt ./checkpoints/swin_tiny_best.pth

# 一次性评估 ./checkpoints/ 下所有已保存模型
python test.py --all
```

输出示例：
```
  Model   : resnet18_baseline
  Test Loss     : 0.8234
  Test Acc (Top-1): 87.45%
  Test Acc (Top-5): 97.82%
```

---

## 实验设置汇总

| 项目 | 配置 |
|------|------|
| 数据集 | 102 Category Flower Dataset |
| 训练集/验证集/测试集 | 1020 / 1020 / 6149（官方划分） |
| 输入尺寸 | 224×224 |
| Batch Size | 32 |
| Epochs | 60 |
| 优化器 | AdamW（所有模型统一） |
| LR (head) | 1e-3 |
| LR (backbone) | 1e-4（CNN）/ 1e-4（ViT）/ 5e-5（Swin） |
| Weight Decay | 1e-2 |
| 调度策略 | Warmup(5 epoch) + Cosine Annealing |
| Loss | CrossEntropyLoss（label_smoothing=0.1） |
| 评价指标 | Top-1 Accuracy、Top-5 Accuracy |
| 数据增强 | RandomResizedCrop、RandomFlip、ColorJitter、RandomRotation |

---

## 可视化（SwanLab 本地模式）

所有训练脚本均以**本地模式**保存日志，无需登录账号：

```bash
# 启动本地可视化面板（训练中或训练后均可）
swanlab watch ./swanlab_logs
# 然后在浏览器打开 http://127.0.0.1:43143
```

每个实验记录以下指标：
- `train/loss`、`train/accuracy`
- `val/loss`、`val/accuracy`
- `test/accuracy`（训练结束后最优 checkpoint 的测试集结果）
- `lr`（当前学习率）

超参数分析的各组实验在同一 project 下，可直接多曲线对比。

---

## 模型权重下载

> 训练完成后将 `./checkpoints/` 上传至网盘，在此填写下载链接。

| 模型 | 下载链接 |
|------|----------|
| ResNet-18 Baseline | _待上传_ |
| ResNet-18 + SE-block | _待上传_ |
| ResNet-18 + CBAM | _待上传_ |
| ViT-Tiny | _待上传_ |
| Swin-Tiny | _待上传_ |

---

## 常见问题

**Q: 下载数据集很慢？**  
可手动下载后放入 `./data/` 目录，torchvision 会自动识别已有文件。

**Q: Swin-T 显存不足？**  
将 `train_swin_t.py` 中 `batch_size` 改为 16，或开启混合精度：
在训练循环中加 `torch.cuda.amp.autocast()`。

**Q: ViT-Tiny 精度低于 ResNet-18？**  
ViT 在小数据集上通常需要更多数据增强或更长训练。可在 `utils.py` 中增强数据增强强度，或将 `num_epochs` 调大至 100。
