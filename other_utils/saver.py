'''
Created on 2019年3月11日

@author: jinglingzhiyu
'''
import torch
import shutil
import os
from PIL import Image
from frame_of_wing.postprocess.gathor import torch2np

def saver_general(root, is_best_acc, is_lowest_loss, state):
#通用模型保存函数
    if os.path.exists(root) is False:
        os.mkdir(root)
    torch.save(state, root + '//' + 'current.pth.tar')
    if is_best_acc is True:
        shutil.copy(root + '//' + 'current.pth.tar', root + '//' + 'best_precision.pth.tar')
    if is_lowest_loss is True:
        shutil.copy(root + '//' + 'current.pth.tar', root + '//' + 'lowest_loss.pth.tar')
    
def saver_imgs(root, imgdict, namelist=None, imgtype='.png', fromnp=False):
#通用文件保存函数,imgdict为数据字典
#函数工作方式:在root下以每个字典的key创建文件夹,并在文件夹内保存key对应的list里面的图片
    if os.path.exists(root) is False:
        os.mkdir(root)
    filetrack = {}
    for filename in imgdict.keys():
        filetrack[filename] = []
        if os.path.exists(os.path.join(root, filename)) is False:
            os.mkdir(os.path.join(root, filename))
    for filename, data in imgdict.items():
        for xx in range(len(data)):
            if namelist is None:
                itemname = str(xx)
            else:
                itemname = namelist[xx]
            if imgtype is not None:
                single_save_root = root + '//' + filename + '//' + itemname + imgtype
            else:
                single_save_root = root + '//' + filename + '//' + itemname
            if fromnp is True:
                data[xx] = Image.fromarray(data[xx])
            data[xx].save(single_save_root)
            filetrack[filename].append(single_save_root)
    return filetrack













