# coding: utf-8
"""
utils.py
增强版：支持多种 CSV 列名（filename/chinese_name/english_name/category_id 等）。
提供 CSVDataset 与 TestImageFolder 以及 accuracy 工具。
"""
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

def default_train_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def default_val_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def _detect_columns(df):
    """
    自动检测 image 列与 label 列，返回 (image_col_name, label_col_name)
    优先级：image: image, filename, file, img, img_name
              label: label, chinese_name, english_name, category, category_id, class
    """
    cols_lower = {c.lower(): c for c in df.columns}
    img_candidates = ['image','filename','file','img','img_name','image_name']
    label_candidates = ['label','chinese_name','english_name','category','category_id','class','categoryname','cat_name']

    image_col = None
    for cand in img_candidates:
        if cand in cols_lower:
            image_col = cols_lower[cand]
            break

    label_col = None
    for cand in label_candidates:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break

    return image_col, label_col

class CSVDataset(Dataset):
    """
    读取 CSV（支持 filename/chinese_name/english_name/category_id 等列），
    image 字段为相对 train_images/ 的文件名或包含子路径。
    """
    def __init__(self, csv_path, img_dir, transform=None, label2idx=None):
        self.df = pd.read_csv(csv_path)
        image_col, label_col = _detect_columns(self.df)
        if image_col is None or label_col is None:
            raise ValueError(f"CSV must contain an image column and a label column. Found: {list(self.df.columns)}")
        self.images = self.df[image_col].astype(str).tolist()
        self.labels_raw = self.df[label_col].astype(str).tolist()
        self.img_dir = img_dir
        self.transform = transform

        if label2idx is None:
            unique = sorted(list(set(self.labels_raw)))
            self.label2idx = {l:i for i,l in enumerate(unique)}
        else:
            self.label2idx = label2idx

        self.targets = [self.label2idx[l] for l in self.labels_raw]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(path):
            # 试一下基名（如果 CSV 里是 class/img.jpg 或含路径）
            alt = os.path.join(self.img_dir, os.path.basename(img_name))
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"Image not found: {path} (tried {alt})")
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx]
        return img, label

class TestImageFolder(Dataset):
    """
    测试集：只包含图片文件的文件夹，返回 (filename, image_tensor)
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Test dir not found: {img_dir}")
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return fname, img

def accuracy(outputs, targets, topk=(1,)):
    """
    outputs: tensor (N, C); targets: tensor (N,)
    返回 top-k 百分比列表
    """
    import torch
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).item())
    return res
