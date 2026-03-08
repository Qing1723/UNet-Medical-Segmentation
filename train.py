import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.dataset import MedicalImageDataset
import os
import numpy as np
from tqdm import tqdm

def dice_coeff(pred, target):
    """计算Dice系数"""
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train(data_dir):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数
    batch_size = 4
    epochs = 20
    lr = 1e-4
    
    # 数据集路径（改成你的实际路径）
    train_image_dir = f'{data_dir}/images/train'
    train_mask_dir = f'{data_dir}/masks/train'
    val_image_dir = f'{data_dir}/images/val'
    val_mask_dir = f'{data_dir}/masks/val'
    
    # 创建数据集
    train_dataset = MedicalImageDataset(train_image_dir, train_mask_dir)
    val_dataset = MedicalImageDataset(val_image_dir, val_mask_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # 二分类交叉熵
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coeff(outputs, masks).item()
        
        # 验证
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_coeff(outputs, masks).item()
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Dice: {train_dice/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Dice: {val_dice/len(val_loader):.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'results/unet_model.pth')
    print("模型已保存")

if __name__ == '__main__':
    train(
        data_dir="C:/复试/我的项目/CV/UNet-Medical-Segmentation/data"
    )