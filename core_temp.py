# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/28

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.init as init  # 导入初始化模块


class CNNWithDropout(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNWithDropout, self).__init__()
        # 1. 第一部分：2个卷积层 + 1个池化层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # 输入为3通道（RGB），输出32通道
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 输入32通道，输出64通道
        self.bn2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 第二个池化层，2x2池化窗口

        # 3. 全连接层部分
        self.fc1 = nn.Linear(64 * 32 * 32, 256)  #   256 * 16 * 16输入为卷积后的扁平化结果，输出512个神经元
        self.fc2 = nn.Linear(256, num_classes)  # 输出层，分类数为 num_classes

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.2)

        # 初始化权重
        self._initialize_weights()  # 添加权重初始化

    def _initialize_weights(self):
        # 遍历模型的每一层，并进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对卷积层进行 He 初始化
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 将偏置初始化为 0
            elif isinstance(m, nn.Linear):  # 对全连接层进行 He 初始化
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.pool1(x)

        # x = F.relu(self.bn3(self.conv3(x)))

        # x = self.pool2(x)

        # 扁平化
        x = torch.flatten(x, 1)  # 使用 flatten 而不是 view

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转10度
    transforms.Resize((64, 64)),  # 根据需要调整图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])










