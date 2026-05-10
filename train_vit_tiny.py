"""
train_vit_tiny.py
───────────────────────────────────────────────────────────────────
Task 1：ViT-Tiny（vit_tiny_patch16_224）via timm
  - 优化器    : AdamW（与所有脚本统一）
  - SwanLab   : 本地模式

ViT-Tiny 规格：Embed=192, Depth=12, Heads=3, Params≈5.7M

Usage:
    pip install timm
    python train_vit_tiny.py
"""

import math
import torch
import torch.nn as nn
import swanlab

try:
    import timm
except ImportError:
    raise ImportError("请先安装 timm：pip install timm")

from utils import get_flower102_loaders, run_training

CFG = {
    "model":           "vit_tiny",
    "timm_name":       "vit_tiny_patch16_224",
    "pretrained":      True,
    "num_classes":     102,
    "img_size":        224,
    "batch_size":      32,
    "num_epochs":      40,
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


def build_model(cfg):
    return timm.create_model(
        cfg["timm_name"], pretrained=cfg["pretrained"],
        num_classes=cfg["num_classes"])


def build_optimizer(model, cfg):
    """
    AdamW 差异化学习率（与 Baseline 完全相同的逻辑）：
      - head（分类层）    → lr_head
      - 骨干（Transformer 主体）→ lr_backbone
    ViT 在 timm 中分类层名为 model.head
    """
    head_params = list(model.head.parameters())
    head_ids    = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    return torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg["lr_backbone"]},
         {"params": head_params,     "lr": cfg["lr_head"]}],
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
    print(f"Using device: {device}  |  timm {timm.__version__}")

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

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,} ({total/1e6:.1f}M)")

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
