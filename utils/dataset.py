import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MedicalImageDataset(Dataset):
    """医学图像分割数据集"""
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        # 读取图片
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
    
        # 读取对应的mask
        # mask文件名可能带 _mask 后缀
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
    
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # 如果找不到mask，创建一个全黑的
            mask = Image.new('L', image.size, 0)
    
        # =====统一尺寸 =====
        target_size = (384,384)  # 统一缩放到选定尺寸
        image = image.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)  # mask用最近邻插值，保持黑白
        # =========================
    
        # 转换为numpy
        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0
    
        # 转换为tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
    
        return image, mask
    
















    