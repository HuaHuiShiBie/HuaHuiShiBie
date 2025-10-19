# coding: utf-8
"""
train.py
兼容 CSV 列名为 filename, category_id, chinese_name, english_name 的情况。
如果没有 val_labels.csv，会从 train_labels.csv 中划分 val，并写出标准化 CSV（image,label）。
运行方式（在 submission/code/ 下）:
python train.py --data_dir ../data --epochs 10 --batch_size 16
"""
import os
import json
import argparse
from tqdm import tqdm
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import create_model
from utils import CSVDataset, default_train_transform, default_val_transform, accuracy

from sklearn.model_selection import train_test_split
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='数据根目录（相对于 code/）')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--save_dir', type=str, default='../model')  # 相对于 code/
    parser.add_argument('--num_workers', type=int, default=0)  # Windows 上默认 0
    parser.add_argument('--pretrained', action='store_true', help='使用 timm pretrained')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='如果没有 val_labels.csv, 从 train_labels.csv 划分验证集比例')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def detect_image_label_columns(df):
    """
    尝试检测图片列与标签列。返回 (image_col_name, label_col_name) 或 (None,None)
    优先寻找 image/filename，然后 label/chinese_name/english_name/category_id。
    """
    cols_lower = {c.lower(): c for c in df.columns}
    img_cands = ['image','filename','file','img','img_name','image_name']
    lbl_cands = ['label','chinese_name','english_name','category','category_id','class','categoryname','cat_name']
    img_col = None
    lbl_col = None
    for c in img_cands:
        if c in cols_lower:
            img_col = cols_lower[c]
            break
    for c in lbl_cands:
        if c in cols_lower:
            lbl_col = cols_lower[c]
            break
    return img_col, lbl_col

def ensure_val_csv(data_dir, val_ratio=0.2, seed=42):
    """
    确保 data_dir 下存在标准化的 train_labels.csv (image,label) 与 val_labels.csv。
    如果 val_labels.csv 已存在且已标准化则直接返回路径。
    否则从 train_labels.csv 读取（支持 filename/chinese_name/english_name/category_id ...），
    生成标准化的 train_labels.csv（覆盖）与 val_labels.csv（写出）。
    返回 (train_csv_path, val_csv_path)
    """
    train_csv = os.path.join(data_dir, 'train_labels.csv')
    val_csv = os.path.join(data_dir, 'val_labels.csv')

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train_labels.csv not found in {data_dir}")

    # 如果已有 val 且已经标准化则直接返回
    if os.path.exists(val_csv):
        dfv = pd.read_csv(val_csv)
        if 'image' in dfv.columns and 'label' in dfv.columns:
            return train_csv, val_csv

    df = pd.read_csv(train_csv)
    img_col, lbl_col = detect_image_label_columns(df)
    if img_col is None or lbl_col is None:
        raise ValueError("train_labels.csv must contain image,label columns (or filename/chinese_name/english_name/category_id). Found: {}".format(list(df.columns)))

    # 取出两列并标准化为 image,label（label 强制为字符串）
    df_std = pd.DataFrame()
    df_std['image'] = df[img_col].astype(str)
    df_std['label'] = df[lbl_col].astype(str)

    # stratified split
    train_df, val_df = train_test_split(df_std, test_size=val_ratio, stratify=df_std['label'], random_state=seed)

    # overwrite train_labels.csv with standardized columns and write val_labels.csv
    train_df.to_csv(train_csv, index=False, encoding='utf-8')
    val_df.to_csv(val_csv, index=False, encoding='utf-8')
    print(f"Standardized and wrote: {train_csv} ({len(train_df)}) and {val_csv} ({len(val_df)})")
    return train_csv, val_csv

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    for images, targets in tqdm(loader, desc='Train', leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        all_outputs.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())
    avg_loss = running_loss / len(loader.dataset)
    import torch
    outputs_cat = torch.cat(all_outputs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    top1 = accuracy(outputs_cat, targets_cat, topk=(1,))[0]
    return avg_loss, top1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Val', leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    avg_loss = running_loss / len(loader.dataset)
    import torch
    outputs_cat = torch.cat(all_outputs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    top1 = accuracy(outputs_cat, targets_cat, topk=(1,))[0]
    return avg_loss, top1

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = os.path.abspath(args.data_dir)
    train_img_dir = os.path.join(data_dir, 'train_images')
    train_csv, val_csv = ensure_val_csv(data_dir, val_ratio=args.val_ratio, seed=args.seed)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_ds = CSVDataset(train_csv, train_img_dir, transform=default_train_transform(args.image_size))
    val_ds = CSVDataset(val_csv, train_img_dir, transform=default_val_transform(args.image_size), label2idx=train_ds.label2idx)

    num_classes = len(train_ds.label2idx)
    print(f"Found {len(train_ds)} train samples, {len(val_ds)} val samples, num_classes={num_classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = create_model(num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using DataParallel on", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_path = os.path.join(save_dir, 'best_model.pth')
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Train loss {train_loss:.4f} acc {train_acc:.2f} | Val loss {val_loss:.4f} acc {val_acc:.2f}")
        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'model_state': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'label2idx': train_ds.label2idx,
                'image_size': args.image_size,
                'model_name': 'convnext_small'
            }
            torch.save(state, best_path)
            idx2label = {int(v):k for k,v in train_ds.label2idx.items()}
            classes = [idx2label[i] for i in range(len(idx2label))]
            cfg = {
                'classes': classes,
                'image_size': args.image_size,
                'model_name': 'convnext_small'
            }
            with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"Saved best model to {best_path} (val acc {best_acc:.2f})")

    print("Training finished. Best val acc: {:.2f}".format(best_acc))

if __name__ == '__main__':
    main()
