import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CSVDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, label2idx=None):
        """
        读取 CSV 格式数据集
        :param csv_path: CSV 文件路径
        :param img_dir: 图片文件夹路径
        :param transform: 图像预处理
        :param label2idx: 可选，标签->索引映射字典（验证/测试阶段需保持一致）
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # 如果没有传入 label2idx，则在训练时自动创建
        if label2idx is None:
            labels = sorted(self.data['label'].unique())
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image']
        label_name = self.data.iloc[idx]['label']

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.label2idx[label_name]

        if self.transform:
            image = self.transform(image)
        return image, label
