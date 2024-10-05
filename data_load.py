# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/10/5

import torch
import torch.nn as nn
import time, datetime
import torch.optim as optim
from core import CNNWithDropout, transform
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import test_folder, train_folder, learning_rate, device, saved_model_path, train_log_file
import logging

# 配置logging模块
logging.basicConfig(
    filename=train_log_file,  # 日志文件名
    filemode='a',             # 追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO        # 日志级别
)


train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)












