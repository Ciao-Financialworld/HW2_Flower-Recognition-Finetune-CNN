"""
train_resnet18_se.py
───────────────────────────────────────────────────────────────────
Task 1 注意力变体：ResNet-18 + SE-block
  - 优化器    : AdamW（与所有脚本统一）
  - SwanLab   : 本地模式
───────────────────────────────────────────────────────────────────
SE-block 原理（Hu et al., CVPR 2018）：
  1. Squeeze : Global Average Pooling → [B, C, 1, 1]
  2. Excite  : FC(C→C/r)→ReLU→FC(C/r→C)→Sigmoid
  3. Scale   : 通道权重 element-wise 乘特征图

Usage:
    python train_resnet18_se.py
"""

import math
import torch
import torch.nn as nn
import torchvision.models as models
import swanlab

from utils import get_flower102_loaders, run_training

CFG = {
    "model":           "resnet18_se",
    "pretrained":      True,
    "num_classes":     102,
    "img_size":        224,
    "batch_size":      32,
    "num_epochs":      40,
    "se_reduction":    16,
    "optimizer":       "AdamW",
    "lr_head":         1e-3,
    "lr_backbone":     1e-4,
    "weight_decay":    1e-2,
    "scheduler":       "warmup_cosine",
    "warmup_epochs":   5,
    "label_smoothing": 0.1,
    "num_workers":     4,
    "data_dir":        "./data",
    "save_dir":        "./checkpoints",
    "log_dir":         "./swanlab_logs",
    "seed":            42,
}


# ── SE-block ──────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.fc(self.pool(x)).view(b, c, 1, 1)
        return x * w


class SEBasicBlock(nn.Module):
    """将 SE 插入 ResNet BasicBlock（BN 后、shortcut 加之前）"""
    expansion = 1

    def __init__(self, block: models.resnet.BasicBlock, reduction: int = 16):
        super().__init__()
        self.conv1      = block.conv1
        self.bn1        = block.bn1
        self.relu       = block.relu
        self.conv2      = block.conv2
        self.bn2        = block.bn2
        self.downsample = block.downsample
        self.stride     = block.stride
        self.se         = SEBlock(block.conv2.out_channels, reduction)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def inject_se(model, reduction):
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, name)
        setattr(model, name,
                nn.Sequential(*[SEBasicBlock(b, reduction) for b in layer]))
    return model


# ── 构建 / 优化器 ──────────────────────────────

def build_model(cfg):
    weights  = models.ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None
    model    = models.resnet18(weights=weights)
    model    = inject_se(model, cfg["se_reduction"])
    model.fc = nn.Linear(model.fc.in_features, cfg["num_classes"])
    return model


def build_optimizer(model, cfg):
    """
    AdamW 参数分组（与 Baseline 逻辑相同，额外将 SE 新参数归入 lr_head 组）：
      - SE 模块（新初始化）  → lr_head
      - fc 层（新初始化）    → lr_head
      - 骨干（预训练权重）   → lr_backbone
    """
    new_ids = set()
    for name, p in model.named_parameters():
        if ".se." in name or "fc." in name:
            new_ids.add(id(p))

    new_params  = [p for p in model.parameters() if id(p)     in new_ids]
    base_params = [p for p in model.parameters() if id(p) not in new_ids]

    return torch.optim.AdamW(
        [{"params": base_params, "lr": cfg["lr_backbone"]},
         {"params": new_params,  "lr": cfg["lr_head"]}],
        weight_decay=cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg):
    def lr_lambda(epoch):
        if epoch < cfg["warmup_epochs"]:
            return float(epoch + 1) / float(cfg["warmup_epochs"])
        progress = (epoch - cfg["warmup_epochs"]) / \
                   max(1, cfg["num_epochs"] - cfg["warmup_epochs"])
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    swanlab.init(
        project         = "flower102-task1",
        experiment_name = CFG["model"],
        config          = CFG,
        mode            = "local",
        logdir          = CFG["log_dir"],
    )

    train_loader, val_loader, test_loader = get_flower102_loaders(
        data_dir=CFG["data_dir"], batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"], img_size=CFG["img_size"],
    )

    model     = build_model(CFG).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    optimizer = build_optimizer(model, CFG)
    scheduler = build_scheduler(optimizer, CFG)

    se_params = sum(p.numel() for n, p in model.named_parameters() if ".se." in n)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}  SE params: {se_params:,}")

    run_training(
        model=model, train_loader=train_loader,
        val_loader=val_loader, test_loader=test_loader,
        optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=device,
        num_epochs=CFG["num_epochs"], run_name=CFG["model"],
        save_dir=CFG["save_dir"],
    )

    swanlab.finish()
    print(f"\n[SwanLab] 日志已保存至 {CFG['log_dir']}/")
    print("[SwanLab] 查看方式：swanlab watch ./swanlab_logs")


if __name__ == "__main__":
    main()
