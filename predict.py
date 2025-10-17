import os
import torch
from PIL import Image
from torchvision import transforms
from models.convnext_model import build_convnext

def predict_single(image_path, model_path="checkpoints/best_model.pth"):
    checkpoint = torch.load(model_path, map_location="cpu")
    label2idx = checkpoint["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}

    num_classes = len(label2idx)
    model = build_convnext(num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()

    print(f"预测类别：{idx2label[pred]}")
    return idx2label[pred]

if __name__ == "__main__":
    predict_single("data/test_images/example.jpg")
