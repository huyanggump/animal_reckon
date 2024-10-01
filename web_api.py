# coding: utf-8
# @Author: WalkerZ
# @Time: 2024/9/28

import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import torch
from reckon_image import reckon_img
from torchvision import transforms
from PIL import Image

# 初始化 Flask 应用
app = Flask(__name__)

# 上传文件保存路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# 检查文件格式
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# 主页面
@app.route('/')
def index():
    return render_template('index.html')


# 文件上传处理
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        result = reckon_img(file)

        return jsonify({'success': True, 'message': f"识别成功！识别结果为，{result}"})

    return jsonify({'success': False, 'message': 'File type not allowed'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=True)










