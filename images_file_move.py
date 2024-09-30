# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/30

import os
import random
import shutil

# 定义文件夹路径
train_folder = "../../data_sets/corrado_animals10/train"
test_folder = "../../data_sets/corrado_animals10/test"

# 读取train文件夹下的子文件夹（类别文件夹）
subfolders = [f.name for f in os.scandir(train_folder) if f.is_dir()]

# 打印子文件夹名称
print("Train子文件夹列表:")
for subfolder in subfolders:
    print(subfolder)

# 创建test文件夹下的对应子文件夹，如果不存在则创建
# for subfolder in subfolders:
#     test_subfolder = os.path.join(test_folder, subfolder)
#     if not os.path.exists(test_subfolder):
#         os.makedirs(test_subfolder)

# 遍历每个子文件夹，随机选取8%的文件并移动到test文件夹
for subfolder in subfolders:
    subfolder_path = os.path.join(train_folder, subfolder)

    print(f"subfolder_path: {subfolder_path}")

    # 获取当前子文件夹中的所有图片文件
    image_files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]

    # 随机选择8%的文件
    sample_size = int(len(image_files) * 0.08)
    selected_files = random.sample(image_files, sample_size)

    # 移动文件到test文件夹中的对应子文件夹
    for file in selected_files:
        src_path = os.path.join(subfolder_path, file)
        dest_path = os.path.join(test_folder, subfolder, file)
        shutil.move(src_path, dest_path)

    print(f"从 {subfolder} 中移动了 {len(selected_files)} 张图片到测试集.")









