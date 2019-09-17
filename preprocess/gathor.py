'''
Created on 2019年2月2日

@author: jinglingzhiyu
'''
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random, os
import numpy as np
import torch
import cv2

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])

# 数据增强：在给定角度中随机进行旋转
class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)

transform_typical=transforms.Compose([transforms.Resize((400, 400)),
                              transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomGrayscale(),
                              # transforms.RandomRotation(20),
                              FixedRotation([0, 90, 180, 270]),
                              transforms.RandomCrop(384),
                              transforms.ToTensor(),
                              normalize,
                              ])
transform_non=transforms.Compose([transforms.Resize((400, 400)),
                                  transforms.ToTensor(),
                                  normalize])

#自己封装的分类任务的増广函数
def transform_typical_cf(img,label,transform = transform_typical):
    return transform(img),label

#自己封装的分类任务非増广函数
def transform_typical_cf_non(img,label,transform = transform_non):
    return transform(img),label

def Synchronizer(data, label, transform_data, transform_label):
#同时对数据图像和label图像做相同的随机变换的方法（同步器）
    new_seed = np.random.randint(1,500)               #定义临时随机种子
    np.random.seed(new_seed)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)
    random.seed(new_seed)
    data = transform_data(data)                       #转换数据
    np.random.seed(new_seed)                          #再次重设相同随机种子,确保对label图像的变换与对data的变换相同
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)
    random.seed(new_seed)
    label = transform_label(label)                    #转换标签
    return data, label

def default_loader(path, mode='RGB'):
    # 默认使用PIL读图
    return Image.open(path).convert(mode)

def imgloader(path=None, imglist=None, convertL=False, tonp=False):
#从文件夹中批量读取图片并保存为list
    if path is None and imglist is None:
        raise Exception('path and imglist cannot be both None')
    elif path is not None and imglist is not None:
        raise Exception('path and imglist cannot be both not None')
    if path is not None:
        imglist = [os.path.join(path, i) for i in os.listdir(path)]
    imgset = []
    for img in imglist:
        img = Image.open(img)
        if convertL is True:
            img = img.convert('L')
        if tonp is True:
            img = np.array(img)
        imgset.append(img)
    return imgset

# 分类任务训练集图片读取
class MyDataset_cf(Dataset):
    def __init__(self, label_list, load_memory=False, transform=None, loader=default_loader, test=False):
        imgs = []
        self.loader = loader
        self.load_memory = load_memory
        self.transform = transform
        self.test = test
        for index, row in label_list.iterrows():
            if self.test is True:
                imgs.append([row['img_path'],None])
            else:
                imgs.append([row['img_path'], row['label']])
            if self.load_memory is True:
                imgs[-1][0] = self.loader(imgs[-1][0])
        self.imgs = imgs

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        if self.load_memory is False:
            img = self.loader(img_path)
        if self.transform is not None:
            img,label = self.transform(img,label)
        if self.test is True:
            return img, img_path
        return img, label

    def __len__(self):
        return len(self.imgs)









