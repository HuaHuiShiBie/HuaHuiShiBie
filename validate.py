import torch
from torchvision import transforms
from utils.dataset_csv import CSVDataset
from models.convnext_model import build_convnext

def evaluate(model_path="checkpoints/best_model.pth", csv_path="data/val_labels.csv", img_dir="data/val_images"):
    checkpoint = torch.load(model_path, map_location="cpu")
    label2idx = checkpoint["label2idx"]
    num_classes = len(label2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_convnext(num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    dataset = CSVDataset(csv_path, img_dir, transform=transform, label2idx=label2idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
    acc = correct / len(dataset)
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
