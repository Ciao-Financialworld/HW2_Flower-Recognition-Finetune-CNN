"""
train_hyperparam_analysis.py
───────────────────────────────────────────────────────────────────
Task 1（2）：超参数分析
  对 ResNet-18（预训练）进行系统性超参数搜索，分析以下变量的影响：
    ① lr_head（分类头学习率）
    ② lr_backbone（骨干学习率）
    ③ batch_size
    ④ 训练 epoch 数
    ⑤ 学习率调度策略（cosine vs step）

  每组实验均通过 SwanLab 本地记录，同一 project 内可直接对比曲线。
  搜索策略：网格搜索，每组完整训练，最后打印汇总表。
───────────────────────────────────────────────────────────────────
Usage:
    python train_hyperparam_analysis.py

  可通过命令行参数指定只跑某一维度：
    python train_hyperparam_analysis.py --analysis lr
    python train_hyperparam_analysis.py --analysis batch
    python train_hyperparam_analysis.py --analysis epoch
    python train_hyperparam_analysis.py --analysis scheduler
    python train_hyperparam_analysis.py --analysis all   # 默认
"""

import argparse
import math
import os
import pprint

import torch
import torch.nn as nn
import torchvision.models as models
import swanlab

from utils import get_flower102_loaders, train_one_epoch, evaluate

# ══════════════════════════════════════════════
#  固定不变的基础配置
# ══════════════════════════════════════════════
BASE = {
    "pretrained":      True,
    "num_classes":     102,
    "img_size":        224,
    "weight_decay":    1e-2,
    "warmup_epochs":   5,
    "label_smoothing": 0.1,
    "num_workers":     4,
    "data_dir":        "./data",
    "save_dir":        "./checkpoints/hpsearch",
    "log_dir":         "./swanlab_logs",
    "seed":            42,
}

# ══════════════════════════════════════════════
#  搜索空间定义
# ══════════════════════════════════════════════

# ① 学习率组合（固定 epoch=20, batch=32）
LR_GRID = [
    {"lr_head": 1e-2, "lr_backbone": 1e-3, "batch_size": 32, "num_epochs": 20,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 20,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-5, "batch_size": 32, "num_epochs": 20,
     "scheduler": "cosine"},
    {"lr_head": 5e-4, "lr_backbone": 5e-5, "batch_size": 32, "num_epochs": 20,
     "scheduler": "cosine"},
]

# ② Batch Size（固定 lr=1e-3/1e-4, epoch=20）
BATCH_GRID = [
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 16, "num_epochs": 20,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 20,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 64, "num_epochs": 20,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 128,"num_epochs": 20,
     "scheduler": "cosine"},
]

# ③ 训练 Epoch 数（固定 lr=1e-3/1e-4, batch=32）
EPOCH_GRID = [
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 20,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 40,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 60,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 80,
     "scheduler": "cosine"},
]

# ④ 学习率调度策略（固定 lr=1e-3/1e-4, batch=32, epoch=20）
SCHEDULER_GRID = [
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 20,
     "scheduler": "cosine"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 20,
     "scheduler": "step"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 20,
     "scheduler": "multistep"},
    {"lr_head": 1e-3, "lr_backbone": 1e-4, "batch_size": 32, "num_epochs": 20,
     "scheduler": "constant"},
]

GRIDS = {
    "lr":        LR_GRID,
    "batch":     BATCH_GRID,
    "epoch":     EPOCH_GRID,
    "scheduler": SCHEDULER_GRID,
}


# ══════════════════════════════════════════════
#  模型 / 优化器 / 调度器构建
# ══════════════════════════════════════════════

def build_model():
    model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, BASE["num_classes"])
    return model


def build_optimizer(model, hp: dict):
    fc_params   = list(model.fc.parameters())
    fc_ids      = {id(p) for p in fc_params}
    base_params = [p for p in model.parameters() if id(p) not in fc_ids]
    return torch.optim.AdamW(
        [{"params": base_params, "lr": hp["lr_backbone"]},
         {"params": fc_params,   "lr": hp["lr_head"]}],
        weight_decay=BASE["weight_decay"],
    )


def build_scheduler(optimizer, hp: dict):
    sched_name = hp["scheduler"]
    n_epochs   = hp["num_epochs"]
    warmup     = BASE["warmup_epochs"]

    if sched_name == "cosine":
        def lr_lambda(e):
            if e < warmup:
                return float(e + 1) / warmup
            prog = (e - warmup) / max(1, n_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_name == "step":
        # 每 20 epoch 衰减 0.1
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    elif sched_name == "multistep":
        # 在 30、50 epoch 各衰减一次
        milestones = [int(n_epochs * 0.5), int(n_epochs * 0.8)]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

    elif sched_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0)

    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


# ══════════════════════════════════════════════
#  单次实验运行
# ══════════════════════════════════════════════

def run_one_experiment(hp: dict, exp_name: str,
                       device: torch.device) -> dict:
    """
    完整训练一个超参数配置，返回 {best_val_acc, test_acc}。
    数据按 hp["batch_size"] 重新创建 DataLoader。
    """
    torch.manual_seed(BASE["seed"])

    train_loader, val_loader, test_loader = get_flower102_loaders(
        data_dir    = BASE["data_dir"],
        batch_size  = hp["batch_size"],
        num_workers = BASE["num_workers"],
        img_size    = BASE["img_size"],
    )

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=BASE["label_smoothing"])
    optimizer = build_optimizer(model, hp)
    scheduler = build_scheduler(optimizer, hp)

    os.makedirs(BASE["save_dir"], exist_ok=True)
    best_val, best_epoch = 0.0, 0

    for epoch in range(1, hp["num_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        swanlab.log({
            "train/loss":     train_loss,
            "train/accuracy": train_acc,
            "val/loss":       val_loss,
            "val/accuracy":   val_acc,
            "lr":             optimizer.param_groups[0]["lr"],
        }, step=epoch)

        print(f"  [{exp_name}] Epoch {epoch:03d}/{hp['num_epochs']}  "
              f"Train={train_acc:.1f}%  Val={val_acc:.1f}%")

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            ckpt = os.path.join(BASE["save_dir"], f"{exp_name}_best.pth")
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict()}, ckpt)

    # 最终 test
    ckpt = torch.load(os.path.join(BASE["save_dir"], f"{exp_name}_best.pth"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc = evaluate(model, test_loader, criterion, device)
    swanlab.log({"test/accuracy": test_acc})

    print(f"  [{exp_name}] Best Val={best_val:.2f}% (ep{best_epoch})  "
          f"Test={test_acc:.2f}%\n")
    return {"best_val": best_val, "test_acc": test_acc, "best_epoch": best_epoch}


# ══════════════════════════════════════════════
#  超参数分析主函数
# ══════════════════════════════════════════════

def run_analysis(analysis_type: str, device: torch.device):
    """
    运行一组超参数网格搜索，在同一 SwanLab project 下记录全部实验。
    """
    grid = GRIDS[analysis_type]
    project_name = f"flower102-hpsearch-{analysis_type}"

    print(f"\n{'═'*62}")
    print(f"  Hyperparameter Analysis: [{analysis_type.upper()}]")
    print(f"  {len(grid)} experiments  →  project: {project_name}")
    print(f"{'═'*62}\n")

    results = []

    for i, hp in enumerate(grid):
        # 实验命名：让曲线在 SwanLab 中一目了然
        if analysis_type == "lr":
            exp_name = (f"lr_head{hp['lr_head']:.0e}"
                        f"_bb{hp['lr_backbone']:.0e}")
        elif analysis_type == "batch":
            exp_name = f"batch{hp['batch_size']}"
        elif analysis_type == "epoch":
            exp_name = f"epoch{hp['num_epochs']}"
        elif analysis_type == "scheduler":
            exp_name = f"sched_{hp['scheduler']}"
        else:
            exp_name = f"exp{i:02d}"

        cfg = {**BASE, **hp, "analysis": analysis_type}
        swanlab.init(
            project         = project_name,
            experiment_name = exp_name,
            config          = cfg,
            mode            = "local",
            logdir          = BASE["log_dir"],
        )

        result = run_one_experiment(hp, exp_name, device)
        results.append({"name": exp_name, "hp": hp, **result})

        swanlab.finish()

    # ── 汇总表 ─────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  [{analysis_type.upper()}] Summary")
    print(f"{'─'*62}")

    if analysis_type == "lr":
        print(f"  {'lr_head':>10} {'lr_backbone':>12} "
              f"{'Best Val':>10} {'Test Acc':>10}")
        print(f"  {'-'*50}")
        for r in results:
            print(f"  {r['hp']['lr_head']:>10.0e} {r['hp']['lr_backbone']:>12.0e}"
                  f"  {r['best_val']:>8.2f}%  {r['test_acc']:>8.2f}%")

    elif analysis_type == "batch":
        print(f"  {'batch_size':>12} {'Best Val':>10} {'Test Acc':>10}")
        print(f"  {'-'*36}")
        for r in results:
            print(f"  {r['hp']['batch_size']:>12}  {r['best_val']:>8.2f}%"
                  f"  {r['test_acc']:>8.2f}%")

    elif analysis_type == "epoch":
        print(f"  {'num_epochs':>12} {'Best Val':>10} {'Test Acc':>10}"
              f"  {'Best Epoch':>12}")
        print(f"  {'-'*48}")
        for r in results:
            print(f"  {r['hp']['num_epochs']:>12}  {r['best_val']:>8.2f}%"
                  f"  {r['test_acc']:>8.2f}%  ep{r['best_epoch']:>4}")

    elif analysis_type == "scheduler":
        print(f"  {'scheduler':>14} {'Best Val':>10} {'Test Acc':>10}")
        print(f"  {'-'*38}")
        for r in results:
            print(f"  {r['hp']['scheduler']:>14}  {r['best_val']:>8.2f}%"
                  f"  {r['test_acc']:>8.2f}%")

    best = max(results, key=lambda x: x["test_acc"])
    print(f"\n  ★ Best config: {best['name']}"
          f"  Test={best['test_acc']:.2f}%")
    print(f"{'─'*62}")

    return results


# ══════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=str, default="all",
                        choices=["lr", "batch", "epoch", "scheduler", "all"],
                        help="要分析的超参数维度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    analyses = list(GRIDS.keys()) if args.analysis == "all" else [args.analysis]

    all_results = {}
    for a in analyses:
        all_results[a] = run_analysis(a, device)

    # ── 全局汇总 ──────────────────────────────
    if len(analyses) > 1:
        print(f"\n{'═'*62}")
        print("  GLOBAL BEST per analysis dimension")
        print(f"{'═'*62}")
        for a, results in all_results.items():
            best = max(results, key=lambda x: x["test_acc"])
            print(f"  [{a:<10}]  best={best['name']}"
                  f"  Test={best['test_acc']:.2f}%")
        print(f"{'═'*62}")

    print(f"\n[SwanLab] 日志保存至 {BASE['log_dir']}/")
    print("[SwanLab] 查看：swanlab watch ./swanlab_logs")


if __name__ == "__main__":
    main()
