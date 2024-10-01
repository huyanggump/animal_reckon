# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/30

import torch
import torch.nn as nn
import time, datetime
import torch.optim as optim
from core import CNNWithDropout, transform
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import test_folder, train_folder, learning_rate, device, saved_model_path
import sys

# 打开一个文件
log_file = open("output.log", "w")

# 将标准输出重定向到文件
sys.stdout = log_file

# 初始化模型、损失函数和优化器
model = CNNWithDropout(num_classes=10)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# 第二步：使用 ImageFolder 加载数据集
train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)

# 查看类别标签与索引的映射关系
print(f"-----train_dataset.class_to_idx: {train_dataset.class_to_idx}")
print(f"-----test_dataset.class_to_idx: {test_dataset.class_to_idx}")


# 第三步：创建数据加载器 (DataLoader)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def train_model_func():
    model.to(device)

    for epoch in range(9):  # 训练9个Epoch
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # print(f"------shape inputs: {inputs.shape}")
            # print(f"------shape labels: {labels.shape}")

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清零梯度

            outputs = model(inputs)  # 前向传播
            # print(f"前向传播 Outputs shape: {outputs.shape}")  # 打印输出的形状
            # print(f"前向传播 Labels shape: {labels.shape}")  # 打印标签的形状
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()
            if i % 10 == 9:  # 每10个batch打印一次损失
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        scheduler.step()  # 调整学习率
    print('\n\n-------------Finished Training-----------\n\n')


def eval_model_func():
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估时不需要计算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')
    torch.save(model.state_dict(), saved_model_path)
    print('\n\n-------------model saved-----------\n\n')


print("\nstart train time: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
train_model_func()
print("\n\nend train time: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("\nstart eval time: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
eval_model_func()
print("\n\nend eval time: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# 记得关闭文件
log_file.close()




