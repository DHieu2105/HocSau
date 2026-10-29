
from pathlib import Path
import argparse
import random
import sys

# Ensure the project root is on sys.path so `models` is importable
# when running directly (e.g. `python scripts/train.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from models.resnet18 import build_resnet18


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "fruit_cls"
CHECKPOINT_DIR = ROOT / "checkpoints"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=eval_transform
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train hoac validation khong co anh.")

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Nhãn train va validation khong khop.")

    if len(train_dataset.classes) < 2:
        raise ValueError("Dataset can it nhat 2 lop.")

    return train_dataset, val_dataset


def build_model(args, num_classes):
    if args.model == "cnn":
        return SimpleCNN(num_classes=num_classes)

    return build_resnet18(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.finetune
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet18"],
        required=True
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("Tham so huan luyen khong hop le.")

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_dataset, val_dataset = build_datasets()
    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_model(args, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=1e-4
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        CHECKPOINT_DIR / f"{args.model}_best.pth"
    )

    best_accuracy = -1.0
    best_loss = float("inf")

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if (
            val_accuracy > best_accuracy
            or (
                val_accuracy == best_accuracy
                and val_loss < best_loss
            )
        ):
            best_accuracy = val_accuracy
            best_loss = val_loss

            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss
                },
                checkpoint_path
            )

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
