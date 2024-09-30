# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/30

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.Resize((64, 64)),  # 根据需要调整图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 第二步：使用 ImageFolder 加载数据集
train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)

# 第三步：创建数据加载器 (DataLoader)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 打印示例: 验证数据集是否正确加载
for images, labels in train_loader:
    print(images.shape)  # 应该输出形状为 [batch_size, channels, height, width]
    print(labels)  # 输出对应的标签





