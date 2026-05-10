"""
ablation_pretrain.py
───────────────────────────────────────────────────────────────────
Task 1 消融实验：预训练权重的影响（Scratch 组）
  - 实验设置 : 随机初始化 → 从零训练（pretrained=False）
  - 对比对象 : train_resnet18_baseline.py（pretrained=True）
  - 唯一变量 : 权重初始化方式。其余所有超参数、优化器（AdamW）、
               随机种子均与 baseline 完全相同，确保公平对比。
  - 可视化   : SwanLab 本地模式（日志写入 ./swanlab_logs/）
───────────────────────────────────────────────────────────────────
Usage:
    python ablation_pretrain.py
"""

import math
import torch
import torch.nn as nn
import torchvision.models as models
import swanlab

from utils import get_flower102_loaders, run_training

# ══════════════════════════════════════════════
#  超参数（与 baseline 完全相同，仅 pretrained=False）
# ══════════════════════════════════════════════
CFG = {
    "model":           "resnet18_scratch",
    "pretrained":      False,
    "num_classes":     102,
    "img_size":        224,
    "batch_size":      32,
    "num_epochs":      40,
    # ── 优化器（AdamW，与所有脚本统一）──
    "optimizer":       "AdamW",
    "lr_head":         1e-3,   # 新 fc 层
    "lr_backbone":     1e-4,   # 骨干（此处为随机初始化）
    "weight_decay":    1e-2,
    # ── 调度器 ──
    "scheduler":       "warmup_cosine",
    "warmup_epochs":   5,
    # ── 其他 ──
    "label_smoothing": 0.1,
    "num_workers":     4,
    "data_dir":        "./data",
    "save_dir":        "./checkpoints",
    "log_dir":         "./swanlab_logs",
    "seed":            42,
}


def build_model(cfg):
    weights  = models.ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None
    model    = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, cfg["num_classes"])
    return model


def build_optimizer(model, cfg):
    """
    AdamW 差异化学习率：
      fc 层      → lr_head
      其余层     → lr_backbone
    与 baseline 保持完全一致的优化器配置，确保唯一变量是初始化方式。
    """
    fc_params       = list(model.fc.parameters())
    fc_ids          = {id(p) for p in fc_params}
    backbone_params = [p for p in model.parameters() if id(p) not in fc_ids]

    return torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg["lr_backbone"]},
         {"params": fc_params,       "lr": cfg["lr_head"]}],
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

    # ── SwanLab 本地初始化 ──
    swanlab.init(
        project         = "flower102-task1-ablation",
        experiment_name = CFG["model"],
        config          = CFG,
        mode            = "local",        # 本地保存，不上传云端
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
    print(f"Init mode   : {'ImageNet pretrained' if CFG['pretrained'] else 'Random init (from scratch)'}")

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
