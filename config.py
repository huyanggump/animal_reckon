# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/29
import torch

# saved_model_path = '../../saved_models/corrado_animals10.pth'
# saved_model_path = '../../saved_models/cifar10.pth'
saved_model_path = '../../saved_models/cifar10_resnet.pth'

# 超参数定义
learning_rate = 0.001 #

# 数据源加载
# train_folder = "../../data_sets/corrado_animals10/train"
# test_folder = "../../data_sets/corrado_animals10/test"
train_folder = "../../data_sets/cifar10/train"
test_folder = "../../data_sets/cifar10/test"

train_log_file = "../../walkerz_logs/walker_animal_reckon/train.log"
api_log_file = "../../walkerz_logs/walker_animal_reckon/web_api.log"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# idx_to_labels = {0: 'cane', 1: 'cavallo', 2: 'elefante', 3: 'farfalla', 4: 'gallina', 5: 'gatto', 6: 'mucca', 7: 'pecora', 8: 'ragno', 9: 'scoiattolo'}
#
# name_translate = {'cane': '狗', 'cavallo': '马', 'elefante': '大象', 'farfalla': '蝴蝶', 'gallina': '鸡', 'gatto': '猫', 'mucca': '牛', 'pecora': '羊', 'ragno': '蜘蛛', 'scoiattolo': '松鼠'}




