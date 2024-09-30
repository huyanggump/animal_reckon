### 1. 使用conda先创建并进入虚拟环境
1. conda create --name xxx_env
2. conda activate xxx_env

### 2. 安装依赖
1. conda install python=3.12.5
2. macos系统下： conda install pytorch::pytorch torchvision torchaudio -c pytorch
   其他系统下请参考：https://pytorch.org/get-started/locally/
3. conda install flask

### 3. 使用数据集
https://www.kaggle.com/datasets/alessiocorrado99/animals10/data
1. 此数据集为jpeg图片格式文件，共有26000张左右的彩色图片，10个动物分类；
2. 此数据集仅有train数据集，需要手动处理划分出test数据集；可使用本代码中的“images_file_move.py”脚本进行处理；
3. 我手动处理后的数据集在此：xxxx.git，test数据集占总图片数量的%8。可直接进行dataset加载并训练；
4. 若要加载jpeg图片文件夹为dataset，可使用datasets.ImageFolder函数操作。



























