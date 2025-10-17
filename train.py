import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.convnext_model import build_convnext
from utils.dataset_csv import CSVDataset
from utils.visualize import plot_training

# ========== 参数 ==========
data_dir = "data"
train_csv = os.path.join(data_dir, "train_labels.csv")
val_csv = os.path.join(data_dir, "val_labels.csv")
train_img_dir = os.path.join(data_dir, "train_images")
val_img_dir = os.path.join(data_dir, "val_images")

num_epochs = 20
batch_size = 16
lr = 1e-4
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 图像增强 ==========
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ========== 数据加载 ==========
train_dataset = CSVDataset(train_csv, train_img_dir, transform=train_transform)
val_dataset = CSVDataset(val_csv, val_img_dir, transform=val_transform, label2idx=train_dataset.label2idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.label2idx)

# ========== 模型 ==========
model = build_convnext(num_classes, pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

best_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

# ========== 训练循环 ==========
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    # 验证阶段
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "label2idx": train_dataset.label2idx
        }, f"{save_dir}/best_model.pth")
        print("✅ 模型已保存")

print(f"训练完成！最佳验证准确率：{best_acc:.4f}")
plot_training(history)
