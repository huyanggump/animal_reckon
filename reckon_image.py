# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/29

import torch
from core import CNNWithDropout, transform
from config import saved_model_path, device, idx_to_labels, name_translate
from PIL import Image

def reckon_img(image):
    model = CNNWithDropout()
    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)
    model.eval()  # 设置模型为推断模式

    # 加载图片并进行预处理
    # image_path = '/Users/walkerz/AI/single_test/cloud111.jpeg'  # 替换为你要分类的图片路径
    # image = Image.open(image_path)  # 使用 PIL 加载图片
    image = Image.open(image)  # 使用 PIL 加载图片
    image = transform(image)  # 预处理图片

    image = image.unsqueeze(0)  # 增加批次维度

    # 将图像输入模型进行分类推断
    with torch.no_grad():
        output = model(image)  # 推断结果
        _, predicted_class = torch.max(output, 1)  # 取最大值对应的分类

    # 打印分类结果
    label = idx_to_labels[predicted_class.item()]
    return f"您上传的动物类别是: {name_translate[label]}"


def reckon_img_test():
    # 加载训练好的模型
    # model = torch.load(saved_model_path)  # 使用你保存的模型文件
    model = CNNWithDropout()
    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)
    model.eval()  # 设置模型为推断模式


    # 加载图片并进行预处理
    image_path = '/Users/walkerz/AI/single_test/cloud111.jpeg'  # 替换为你要分类的图片路径
    image = Image.open(image_path)          # 使用 PIL 加载图片
    image = transform(image)               # 预处理图片
    image = image.unsqueeze(0)              # 增加批次维度

    # 将图像输入模型进行分类推断
    with torch.no_grad():
        output = model(image)               # 推断结果
        _, predicted_class = torch.max(output, 1)  # 取最大值对应的分类

    # 打印分类结果
    label = idx_to_labels[predicted_class.item()]
    print(f"您上传的动物类别是: {name_translate[label]}")
    # print(f"Predicted class: {predicted_class}")





