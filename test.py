"""
test.py
───────────────────────────────────────────────────────────────────
对已保存的模型 checkpoint 在测试集上进行评估。
支持所有 Task 1 训练的模型：
  resnet18_baseline / resnet18_se / resnet18_cbam / vit_tiny / swin_tiny

Usage:
    # 评估 baseline
    python test.py --model resnet18_baseline \
                   --ckpt ./checkpoints/resnet18_baseline_best.pth

    # 评估 SE
    python test.py --model resnet18_se \
                   --ckpt ./checkpoints/resnet18_se_best.pth

    # 评估 ViT-Tiny
    python test.py --model vit_tiny \
                   --ckpt ./checkpoints/vit_tiny_best.pth

    # 一次性评估所有已保存的 checkpoint
    python test.py --all
───────────────────────────────────────────────────────────────────
"""

import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models

from utils import get_flower102_loaders, evaluate

# ── 每个模型的构建函数（不加载预训练，只定义结构）──────────────
NUM_CLASSES = 102


def build_resnet18_baseline():
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_resnet18_se():
    """SE-block 版本（从 train_resnet18_se 导入注入逻辑）"""
    from train_resnet18_se import build_model
    return build_model({"pretrained": False,
                        "num_classes": NUM_CLASSES,
                        "se_reduction": 16})


def build_resnet18_cbam():
    from train_resnet18_cbam import build_model
    return build_model({"pretrained": False,
                        "num_classes": NUM_CLASSES,
                        "cbam_reduction": 16,
                        "cbam_kernel": 7})


def build_vit_tiny():
    try:
        import timm
    except ImportError:
        raise ImportError("pip install timm")
    return timm.create_model("vit_tiny_patch16_224",
                              pretrained=False, num_classes=NUM_CLASSES)


def build_swin_tiny():
    try:
        import timm
    except ImportError:
        raise ImportError("pip install timm")
    return timm.create_model("swin_tiny_patch4_window7_224",
                              pretrained=False, num_classes=NUM_CLASSES)


MODEL_BUILDERS = {
    "resnet18_baseline": build_resnet18_baseline,
    "resnet18_se":       build_resnet18_se,
    "resnet18_cbam":     build_resnet18_cbam,
    "vit_tiny":          build_vit_tiny,
    "swin_tiny":         build_swin_tiny,
}

# 消融实验的 checkpoint 也用 baseline 结构
MODEL_BUILDERS["resnet18_pretrained"] = build_resnet18_baseline
MODEL_BUILDERS["resnet18_scratch"]    = build_resnet18_baseline


# ── 单个模型评估 ────────────────────────────────────────────────

def test_one(model_name: str, ckpt_path: str,
             device: torch.device,
             data_dir: str = "./data",
             batch_size: int = 64):

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"未知模型名: {model_name}\n"
                         f"可选: {list(MODEL_BUILDERS.keys())}")

    if not os.path.exists(ckpt_path):
        print(f"  [跳过] checkpoint 不存在: {ckpt_path}")
        return None

    # 构建模型并加载权重
    model = MODEL_BUILDERS[model_name]().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    saved_epoch   = ckpt.get("epoch", "?")
    saved_val_acc = ckpt.get("val_acc", float("nan"))

    print(f"\n  Model   : {model_name}")
    print(f"  Ckpt    : {ckpt_path}")
    print(f"  Saved   : epoch={saved_epoch}, val_acc={saved_val_acc:.2f}%")

    # 数据
    _, _, test_loader = get_flower102_loaders(
        data_dir=data_dir, batch_size=batch_size,
        num_workers=4, img_size=224)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Top-5 accuracy
    top5 = top5_accuracy(model, test_loader, device)

    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Acc (Top-1): {test_acc:.2f}%")
    print(f"  Test Acc (Top-5): {top5:.2f}%")

    return {"model": model_name, "test_acc": test_acc,
            "top5": top5, "test_loss": test_loss}


@torch.no_grad()
def top5_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, pred = outputs.topk(5, dim=1)          # [B, 5]
        correct += pred.eq(targets.view(-1, 1).expand_as(pred)).any(dim=1).sum().item()
        total   += targets.size(0)
    return 100.0 * correct / total


# ── 一次性评估所有 checkpoint ───────────────────────────────────

def test_all(ckpt_dir: str = "./checkpoints",
             data_dir:  str = "./data",
             device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []
    for model_name in MODEL_BUILDERS:
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
        result = test_one(model_name, ckpt_path, device, data_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n未找到任何 checkpoint，请先完成训练。")
        return

    # 汇总表
    print(f"\n{'='*60}")
    print(f"  ALL MODELS — Test Set Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<24} {'Top-1 Acc':>10} {'Top-5 Acc':>10}")
    print(f"  {'-'*46}")
    for r in sorted(all_results, key=lambda x: -x["test_acc"]):
        print(f"  {r['model']:<24} {r['test_acc']:>9.2f}%  {r['top5']:>9.2f}%")
    print(f"{'='*60}")


# ── 入口 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on Flowers-102 test set")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_BUILDERS.keys()),
                        help="模型名称")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint 路径（.pth 文件）")
    parser.add_argument("--all", action="store_true",
                        help="自动评估 ./checkpoints/ 下所有已保存的模型")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints",
                        help="--all 模式下 checkpoint 所在目录")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all:
        test_all(ckpt_dir=args.ckpt_dir, data_dir=args.data_dir, device=device)
    elif args.model and args.ckpt:
        test_one(args.model, args.ckpt, device, args.data_dir, args.batch_size)
    else:
        parser.print_help()
        print("\n示例：")
        print("  python test.py --model resnet18_baseline "
              "--ckpt ./checkpoints/resnet18_baseline_best.pth")
        print("  python test.py --all")


if __name__ == "__main__":
    main()
