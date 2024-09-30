# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/29

import torch
from core import CNNWithDropout
from config import saved_model_path, device

model = CNNWithDropout()
model.load_state_dict(torch.load(saved_model_path))
model.to(device)





