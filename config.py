# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/29
import torch

saved_model_path = '../../saved_models/corrado_animals10.pth'

# 超参数定义
learning_rate = 0.001

# 数据源加载
train_folder = "../../data_sets/corrado_animals10/train"
test_folder = "../../data_sets/corrado_animals10/test"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

