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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)  # 输入为3通道（RGB），输出32通道
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1)  # 输入32通道，输出64通道
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 第一个池化层，2x2池化窗口

        # 2. 第二部分：2个卷积层 + 1个池化层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1)  # 输入64通道，输出128通道
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=1)  # 输入128通道，输出256通道
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 第二个池化层，2x2池化窗口

        # 3. 全连接层部分
        self.fc1 = nn.Linear(256 * 13 * 13, 512)  #   256 * 16 * 16输入为卷积后的扁平化结果，输出512个神经元
        self.fc2 = nn.Linear(512, 128)  # 输出128个神经元
        self.fc3 = nn.Linear(128, num_classes)  # 输出层，分类数为 num_classes

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.3)

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
        # 第一部分：2个卷积层 + 池化层
        x = F.relu(self.conv1(x))  # 第一层卷积+激活
        x = F.relu(self.conv2(x))  # 第二层卷积+激活
        x = self.pool1(x)  # 池化层

        # 第二部分：2个卷积层 + 池化层
        x = F.relu(self.conv3(x))  # 第三层卷积+激活
        x = F.relu(self.conv4(x))  # 第四层卷积+激活
        x = self.pool2(x)  # 池化层

        # 扁平化，将卷积输出展开为一维向量
        x = torch.flatten(x, 1)

        # 全连接层部分
        x = F.relu(self.fc1(x))  # 第一个全连接层
        x = self.dropout(x)  # 加入 Dropout 防止过拟合
        x = F.relu(self.fc2(x))  # 第二个全连接层
        x = self.dropout(x)
        x = self.fc3(x)  # 输出层（没有激活函数，因为后续会使用 softmax 或交叉熵）

        return x


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转10度
    transforms.Resize((64, 64)),  # 根据需要调整图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])










