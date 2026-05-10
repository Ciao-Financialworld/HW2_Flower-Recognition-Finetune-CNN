"""
utils.py — 公共工具：数据加载、训练/评估循环
102 Category Flower Dataset

SwanLab 本地保存说明：
  所有脚本统一使用 mode="local"，日志写入 ./swanlab_logs/<experiment_name>/
  无需登录，无需联网，直接用 `swanlab watch ./swanlab_logs` 本地查看。
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 统一使用 Hugging Face 镜像，供 timm / huggingface_hub 拉取预训练权重时使用。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ─────────────────────────────────────────────
#  数据集加载
# ─────────────────────────────────────────────

def get_flower102_loaders(data_dir: str = "./data",
                          batch_size: int = 32,
                          num_workers: int = 4,
                          img_size: int = 224):
    """
    返回 Flowers102 的 train / val / test DataLoader。
    Flowers102: 102 类花卉，train=1020, val=1020, test=6149
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),   # 256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.Flowers102(
        root=data_dir, split="train",
        transform=train_transform, download=True)
    val_set = torchvision.datasets.Flowers102(
        root=data_dir, split="val",
        transform=val_transform, download=True)
    test_set = torchvision.datasets.Flowers102(
        root=data_dir, split="test",
        transform=val_transform, download=True)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_set,  shuffle=False, **loader_kw)

    print(f"[Dataset] train={len(train_set)}  val={len(val_set)}  "
          f"test={len(test_set)}  classes=102")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  训练 / 评估一个 epoch
# ─────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


# ─────────────────────────────────────────────
#  通用训练主循环（供各脚本调用）
# ─────────────────────────────────────────────

def run_training(model, train_loader, val_loader, test_loader,
                 optimizer, scheduler, criterion, device,
                 num_epochs, run_name, save_dir="./checkpoints"):
    """
    完整训练循环，自动记录 SwanLab 指标，保存最优 checkpoint。
    """
    import swanlab

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc, best_epoch = 0.0, 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        # ── SwanLab 记录 ──
        swanlab.log({
            "train/loss":     train_loss,
            "train/accuracy": train_acc,
            "val/loss":       val_loss,
            "val/accuracy":   val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"Epoch [{epoch:03d}/{num_epochs}]  "
              f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.2f}%  "
              f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.2f}%")

        # ── 保存最优模型 ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            ckpt_path    = os.path.join(save_dir, f"{run_name}_best.pth")
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc}, ckpt_path)
            print(f"  ✔ New best val acc={val_acc:.2f}% — saved to {ckpt_path}")

    # ── 最终 Test 评估 ──
    best_ckpt = torch.load(os.path.join(save_dir, f"{run_name}_best.pth"),
                           map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    swanlab.log({"test/accuracy": test_acc, "test/loss": test_loss})
    print(f"\n{'='*60}")
    print(f"[{run_name}]  Best Val Acc={best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"[{run_name}]  Test  Acc   ={test_acc:.2f}%")
    print(f"{'='*60}\n")

    return best_val_acc, test_acc
