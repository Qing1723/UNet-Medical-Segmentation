import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from models.unet import UNet

def predict_single_image(model, image_path, device):
    """预测单张图片"""
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 预处理
    image_np = np.array(image.resize((256, 256))) / 255.0
    image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        mask = model(image_tensor)
    
    # 后处理
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='Reds')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    plt.savefig('results/prediction.png')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load('results/unet_model.pth', map_location=device))
    
    # 预测示例
    test_image = 'data/images/val/legionella-pneumonia-2.jpg'
    predict_single_image(model, test_image, device)
    test_image = 'data/images/val/MERS-CoV-1-s2.0-S0378603X1500248X-gr4e.jpg'
    predict_single_image(model, test_image, device)
    test_image = 'data/images/val/pneumocystis-jiroveci-pneumonia-2.png'
    predict_single_image(model, test_image, device)
if __name__ == '__main__':
    main()