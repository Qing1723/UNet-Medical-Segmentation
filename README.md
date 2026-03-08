# U-Net 医学图像分割系统

使用PyTorch实现U-Net网络，在肺部CT图像上进行病灶区域分割，Dice系数达到93.9%。

## 实验结果

| 分辨率 | 训练集Dice | 验证集Dice | 轮数 |
|--------|------------|------------|------|
| 256×256 | 95.5% | 93.9% | 10 |
| 384×384 | 95.4% | 93.4% | 20 |

- 高分辨率下仍保持93.4%的Dice，证明模型鲁棒性
- 训练集与验证集差距仅1.6%，无过拟合

### 分割效果示例
查看results中示例

## 环境要求

Python 3.8+
PyTorch 1.9+
其他依赖见 requirements.txt

## 数据集准备
下载COVID-19 CT影像数据集

将图片放在 data/images/train 和 data/images/val

将对应的mask放在 data/masks/train 和 data/masks/val
可自行运行DataProcessing.py对原数据集进行数据处理，得到训练所需数据集
