
import argparse
from pathlib import Path
import sys

# Ensure the project root is on sys.path so `models` is importable
# when running directly (e.g. `python scripts/predict.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.cnn import SimpleCNN
from models.resnet18 import build_resnet18


def load_model(model_name, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    if model_name == "cnn":
        model = SimpleCNN(num_classes=len(class_names))
    else:
        model = build_resnet18(
            num_classes=len(class_names),
            pretrained=False,
            freeze_backbone=False
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def predict_image(model, class_names, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    print(f"Image: {image_path}")
    print(f"Predicted class: {class_names[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item() * 100:.2f}%")
    print("Class probabilities:")

    for index, class_name in enumerate(class_names):
        print(f"{class_name}: {probabilities[index].item() * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("checkpoints") / f"{args.model}_best.pth"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, class_names = load_model(
        args.model,
        checkpoint_path,
        device
    )

    predict_image(model, class_names, image_path, device)


if __name__ == "__main__":
    main()
