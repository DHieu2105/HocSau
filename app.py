from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.cnn import SimpleCNN
from models.resnet18 import build_resnet18


st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍑",
    layout="centered"
)


@st.cache_resource
def load_model(model_name):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
        Path(__file__).parent
        / "checkpoints"
        / f"{model_name}_best.pth"
    )

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

    return model, class_names, device


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)


st.title("🍑 Fruit Classifier")
st.write("Phân loại ảnh đào và bưởi bằng mô hình học sâu.")

model_choice = st.selectbox(
    "Chọn mô hình",
    ["CNN", "ResNet18"]
)

model_name = {
    "CNN": "cnn",
    "ResNet18": "resnet18"
}[model_choice]

uploaded_file = st.file_uploader(
    "Tải ảnh trái cây lên",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Ảnh đầu vào",
        use_container_width=True
    )

    if st.button("Phân loại ảnh", type="primary"):
        try:
            model, class_names, device = load_model(model_name)
            image_tensor = preprocess_image(image).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]

            predicted_index = probabilities.argmax().item()
            predicted_class = class_names[predicted_index]
            confidence = probabilities[predicted_index].item()

            st.success(f"Kết quả: {predicted_class}")
            st.metric(
                "Độ tin cậy của dự đoán",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Xác suất theo từng lớp")

            for index, class_name in enumerate(class_names):
                probability = probabilities[index].item()

                st.write(
                    f"{class_name}: {probability * 100:.2f}%"
                )
                st.progress(probability)

            st.caption(
                "Xác suất là đầu ra của mô hình, "
                "không phải bảo đảm về độ chính xác thực tế."
            )

        except FileNotFoundError:
            st.error(
                "Không tìm thấy checkpoint. "
                "Hãy kiểm tra thư mục checkpoints."
            )
