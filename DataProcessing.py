import os
import shutil
from PIL import Image
import numpy as np

def prepare_dataset(data_dir, output_dir):
    """
    使用 lungVAE-masks 里的现成mask（文件名多 _mask）
    """
    # 创建输出文件夹
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/masks/train', exist_ok=True)
    os.makedirs(f'{output_dir}/masks/val', exist_ok=True)
    
    # 原始图片文件夹
    img_dir = f'{data_dir}/images'
    # mask文件夹
    mask_dir = f'{data_dir}/annotations/lungVAE-masks'
    
    # 获取所有mask文件
    mask_files = os.listdir(mask_dir)
    print(f"找到 {len(mask_files)} 个mask文件")
    
    # 找到同时有图片和mask的配对
    items = []
    for mask_file in mask_files:
        # mask文件名格式: xxx_mask.png
        if not mask_file.endswith('_mask.png'):
            continue
            
        # 去掉 _mask.png 得到基础名
        base_name = mask_file.replace('_mask.png', '')
        
        # 尝试常见的图片格式
        for ext in ['.jpg', '.jpeg', '.png']:
            img_file = base_name + ext
            img_path = os.path.join(img_dir, img_file)
            
            if os.path.exists(img_path):
                items.append({
                    'img_path': img_path,
                    'mask_path': os.path.join(mask_dir, mask_file),
                    'img_filename': img_file,
                    'mask_filename': mask_file
                })
                break
    
    print(f"找到 {len(items)} 张图片-mask配对")
    
    if len(items) == 0:
        print("错误：没有找到任何配对")
        # 打印几个mask文件名示例
        print("\nmask文件名示例:")
        for f in mask_files[:5]:
            print(f"  {f}")
        return
    
    # 划分训练集和验证集
    split_idx = int(len(items) * 0.8)
    
    # 复制训练集
    for i, item in enumerate(items[:split_idx]):
        # 复制图片
        dst_img = f'{output_dir}/images/train/{item["img_filename"]}'
        shutil.copy(item['img_path'], dst_img)
        
        # 复制mask（保持原名）
        dst_mask = f'{output_dir}/masks/train/{item["mask_filename"]}'
        shutil.copy(item['mask_path'], dst_mask)
    
    # 复制验证集
    for i, item in enumerate(items[split_idx:]):
        shutil.copy(item['img_path'], 
                   f'{output_dir}/images/val/{item["img_filename"]}')
        shutil.copy(item['mask_path'], 
                   f'{output_dir}/masks/val/{item["mask_filename"]}')
    
    print(f"\n数据集准备完成！")
    print(f"训练集: {split_idx}张")
    print(f"验证集: {len(items)-split_idx}张")
    
    # 检查第一张mask
    if len(items) > 0:
        mask = Image.open(items[0]['mask_path'])
        print(f"\n示例mask信息:")
        print(f"  文件名: {items[0]['mask_filename']}")
        print(f"  对应图片: {items[0]['img_filename']}")
        print(f"  尺寸: {mask.size}")
        print(f"  模式: {mask.mode}")
        
        # 统计白色像素
        mask_array = np.array(mask)
        white_pixels = np.sum(mask_array > 0)
        total_pixels = mask_array.size
        print(f"  白色像素: {white_pixels} ({white_pixels/total_pixels*100:.2f}%)")
        
        if white_pixels == 0:
            print("  ⚠️ 警告：mask全黑！")
        else:
            print("  ✅ mask有白色区域")

if __name__ == '__main__':
    prepare_dataset(
        data_dir='C:/复试/我的项目/CV/covid-chestxray-dataset',
        output_dir='covid_segmentation'
    )