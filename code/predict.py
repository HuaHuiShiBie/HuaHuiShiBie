# coding: utf-8
"""
predict.py
读取 ../model/config.json 与 ../model/best_model.pth，读取 ../test_dataset/ 下图片，
输出 ../results/submission.csv（image,label）
"""
import os
import argparse
import json
import csv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import create_model
from utils import TestImageFolder, default_val_transform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='../test_dataset')
    parser.add_argument('--model_path', type=str, default='../model/best_model.pth')
    parser.add_argument('--config_path', type=str, default='../model/config.json')
    parser.add_argument('--results_dir', type=str, default='../results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_checkpoint(model_path, device, num_classes, model_name='convnext_small'):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(num_classes=num_classes, pretrained=False, model_name=model_name)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    return model

def collate_fn(batch):
    fnames = [x[0] for x in batch]
    images = torch.stack([x[1] for x in batch], dim=0)
    return fnames, images

def main():
    args = parse_args()
    test_dir = os.path.abspath(args.test_dir)
    model_path = os.path.abspath(args.model_path)
    config_path = os.path.abspath(args.config_path)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    submission_path = os.path.join(results_dir, 'submission.csv')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}. Run train.py first.")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    classes = cfg.get('classes', None)
    if classes is None:
        raise ValueError("config.json must contain 'classes' key")
    num_classes = len(classes)
    model_name = cfg.get('model_name', 'convnext_small')
    image_size = cfg.get('image_size', 224)

    dataset = TestImageFolder(test_dir, transform=default_val_transform(image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = load_checkpoint(model_path, device, num_classes=num_classes, model_name=model_name)

    results = []
    with torch.no_grad():
        for fnames, images in tqdm(loader, desc='Predict'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist()
            for fname, p in zip(fnames, preds):
                label = classes[p]
                results.append((fname, label))

    with open(submission_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image','label'])
        for fn, lb in results:
            writer.writerow([fn, lb])

    print(f"Saved {len(results)} predictions to {submission_path}")

if __name__ == '__main__':
    main()
