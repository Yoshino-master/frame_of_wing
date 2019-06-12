'''
Created on 2019年3月7日

@author: jinglingzhiyu
'''
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from frame_of_wing.preprocess.gathor import default_loader, Synchronizer, FixedRotation
import numpy as np
from matplotlib import pylab
import torchvision
import torch

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

#建议Resize大小:286
transform_cGAN_train_data = transforms.Compose([transforms.Resize((256,256)),
#                                                 transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                                                #transforms.RandomHorizontalFlip(),
#                                                 transforms.RandomGrayscale(),
                                                #FixedRotation([0, 90, 180, 270]),
                                                transforms.RandomCrop(224),
                                                transforms.ToTensor(),
                                                normalize
                                                ])

transform_cGAN_train_label = transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                #transforms.RandomHorizontalFlip(),
#                                                 transforms.RandomGrayscale(),
                                                #FixedRotation([0, 90, 180, 270]),
                                                transforms.RandomCrop(224),
                                                transforms.ToTensor(),
                                                normalize
                                                ])

transform_cGAN_test = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          normalize])

def get_noope_transform(resize_shape):
    return transforms.Compose([transforms.Resize(resize_shape),
                               transforms.ToTensor(),
                               normalize])


def temp_change_img(img):
    out = np.zeros((*img.shape[1:],3))
    for xx in range(3):
        out[:,:,xx] = img[xx,:,:]
    return out


#cGAN训练集图片读取
class MyDataset_cGAN(Dataset):
    def __init__(self, label_list, transform_data, transform_label, test=False, need_show=False):
        imgs = []
        self.loader = default_loader
        self.transform_data = transform_data
        self.transform_label=transform_label
        self.test = test
        self.need_show = need_show
        self.Synchronizer = Synchronizer
        for index, row in label_list.iterrows():
            if self.test is False:
                imgs.append([row['img_path'], row['label_path']])
        self.imgs = imgs
    
    def __getitem__(self, index):
        img_path, label_path = self.imgs[index]
        img, label = self.loader(img_path), self.loader(label_path, 'L')
        h, w = img.size[0], img.size[1]
        if self.need_show is True:
            pylab.imshow(label)
            pylab.show()
            pylab.imshow(img)
            pylab.show()
        img, label = self.Synchronizer(img, label, self.transform_data, self.transform_label)
        return img, label, {'img_path' : img_path,
                            'label_path': label_path,
                            'h' : h, 'w' : w }
    
    def __len__(self):
        return len(self.imgs)




if __name__ =='__main__':
    temp_path = r'D:\精灵之羽\羽\科大线\190225\others\pix2pix-tensorflow-master\facades\train\1.jpg'
    from PIL import Image
    from matplotlib import pylab
    import copy
    import torch
    img = Image.open(temp_path).convert('RGB')
    #img1 = copy.deepcopy(img)
    img = transform_cGAN_train_data(img)
    print(img.shape)
    print(torch.mean(img))
    print(torch.min(img))
    print(torch.max(img))
    

