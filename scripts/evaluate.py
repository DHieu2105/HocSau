
from pathlib import Path
import argparse
import sys

# Ensure the project root is on sys.path so `models` is importable
# when running directly (e.g. `python scripts/evaluate.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from models.resnet18 import build_resnet18


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "datasets" / "fruit_cls" / "test"
CHECKPOINT_DIR = ROOT / "checkpoints"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(model_name, checkpoint, device):
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    if model_name == "cnn":
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = build_resnet18(
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def evaluate(model, loader, criterion, num_classes, device):
    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64
    )

    total_loss = 0.0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            for actual, predicted in zip(
                labels.cpu(), predictions.cpu()
            ):
                confusion[actual.item(), predicted.item()] += 1

    accuracy = confusion.diag().sum().item() / max(total, 1)
    average_loss = total_loss / max(total, 1)

    return confusion, accuracy, average_loss


def print_metrics(confusion, class_names):
    print("\nCONFUSION MATRIX")
    print("Rows: Actual | Columns: Predicted")
    print(f"{'':>12}" + "".join(
        f"{name:>12}" for name in class_names
    ))

    for index, name in enumerate(class_names):
        values = "".join(
            f"{confusion[index, j].item():>12}"
            for j in range(len(class_names))
        )
        print(f"{name:>12}{values}")

    print("\nCLASSIFICATION REPORT")
    print(
        f"{'Class':<12}{'Precision':>12}"
        f"{'Recall':>12}{'F1-score':>12}{'Support':>12}"
    )

    for index, name in enumerate(class_names):
        tp = confusion[index, index].item()
        fp = confusion[:, index].sum().item() - tp
        fn = confusion[index, :].sum().item() - tp
        support = confusion[index, :].sum().item()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print(
            f"{name:<12}{precision:>12.4f}"
            f"{recall:>12.4f}{f1:>12.4f}{support:>12}"
        )

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    print(f"\nTotal test images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet18"],
        required=True
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
        CHECKPOINT_DIR / f"{args.model}_best.pth"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint: {checkpoint_path}"
        )

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Khong tim thay test dataset: {TEST_DIR}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform
    )

    if len(test_dataset) == 0:
        raise ValueError("Test dataset khong co anh.")

    model, class_names = load_model(
        args.model,
        checkpoint,
        device
    )

    if test_dataset.classes != class_names:
        raise ValueError(
            f"Nhãn test {test_dataset.classes} "
            f"khong khop checkpoint {class_names}"
        )

    loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    criterion = nn.CrossEntropyLoss()

    confusion, accuracy, loss = evaluate(
        model,
        loader,
        criterion,
        len(class_names),
        device
    )

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    print_metrics(confusion, class_names)


if __name__ == "__main__":
    main()
