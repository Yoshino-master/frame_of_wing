'''
Created on 2019年2月9日

@author: jinglingzhiyu
'''
import torchvision
import numpy as np
import torchvision.transforms as transforms
from frame_of_wing.other_utils.recorder import method_AverageMeter



def get_itransform(shape=None):
    #将tensor转换为PIL文件
    trans = [transforms.ToPILImage()]
    if shape is not None:
        trans.append(transforms.Resize(shape))
    return transforms.Compose(trans)

itrans_vanilla = [get_itransform(), np.array]

class general_trans():
    #通用转换类
    def __init__(self, translist=itrans_vanilla):
        self.translist = translist
    def __call__(self, data):
        for trans in self.translist:
            data = trans(data)
        return data

class estimator():
    #集成的指标估计器
    #注意:
    #1.本类别依赖于method_AverageMeter方法,即recorders中只对method_AverageMeter类起作用
    def __init__(self, m_recorders):
        self.recorders = m_recorders
    def updates(self, data):
        for keyname, recorder in self.recorders.items():
            if(isinstance(recorder, method_AverageMeter)):
                recorder._updates(data)
    def update(self, data, n=1):
        for keyname, recorder in self.recorders.items():
            if(isinstance(recorder, method_AverageMeter)):
                recorder._update(data, n=n)

def my_torch2imgs(images, trans=torchvision.transforms.ToPILImage(), mode='tanh'):
    imgset = []
    for xx in range(images.shape[0]):
        if(mode == 'tanh'):
            img = (images[xx] + 1.0) / 2.0
        imgset.append(trans(img))
    return imgset

def torch2np(img):
    out = np.zeros((*img.shape[1:],3))
    for xx in range(3):
        out[:,:,xx] = img[xx,:,:]
    return out




if __name__ == '__main__':
    import torch
    import numpy as np
    data = torch.ones(3, 256, 256)
    trans = [get_itransform(), np.array]
    transformer = general_trans(trans)
    out = transformer(data)
    print(out.shape)
    

