'''
Created on 2019年2月2日

@author: jinglingzhiyu
'''
import torch
import numpy as np
import random
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
from PIL import Image
import pylab

tempdict = {'a':[0,1,3,9,7], 'd':[23,6]}

def _main():
    root = r'D:\精灵之羽\羽\科大线\190225\others\pix2pix-tensorflow-master\facades_test_v1\index.html'
    with open(root, 'r') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    for a in tempdict.keys():
        print(tempdict[a])

def _main2():
    import sys
    sys.path.append(r'D:\精灵之羽\羽\科大线\190225\others\pytorch-CycleGAN-and-pix2pix-master')
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    from data import create_dataset
    from models import create_model
    opt = TrainOptions().parse()
    model = create_model(opt)
    #model.setup(opt)
    model.eval()
    data = torch.ones(1,6,256,256)
    out = model.netD(data)
    print(out)
    from frame_of_wing.my_model.conditionGAN_2 import discriminator
    model2 = discriminator()
    print(model2)
    model2.eval()
    out2 = model2(data)
    print(out2)

def _main3():
    import sys
    sys.path.append(r'D:\精灵之羽\羽\科大线\190225\others\pytorch-CycleGAN-and-pix2pix-master')
    from options.train_options import TrainOptions
    from frame_of_wing.preprocess.gathor import set_random_seed
    opt = TrainOptions().parse()
    from data import create_dataset
    from models import create_model
    opt = TrainOptions().parse()
    set_random_seed(10)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    image = torch.ones(1,3,256,256)
    label = torch.ones(1,3,256,256)
    data = {'A':label, 'B':image, 'B_paths':'你开心就好'}
    model.set_input(data)
    model.optimize_parameters()
    
def _main4():
    from tensorboardX import SummaryWriter
    from tensorflow.summary import FileWriter, merge_all
    import time
    #wasgasgssf = merge_all()
    writer1 = SummaryWriter(r'C:\Users\jinglingzhiyu\Desktop\temp\tfshow\1')
    #writer2 = SummaryWriter(r'C:\Users\jinglingzhiyu\Desktop\temp\tfshow\2')
    for i in range(500):
        writer1.add_scalars('test1', {'train' : np.sin(i*0.1),
                                      'test' : np.cos(i*0.1)}, i)
        #writer2.add_scalar('train', np.cos(i*0.1), i)
        time.sleep(1)
    writer1.close()
    #writer2.close()

def split_saliency(img, n=3, max_n=6):
    splited = []
    big = (img.astype('int32') * n)
    for i in range(n - 1):
        splited.append((img / (2 ** (max_n - i))).astype('int32'))
        big = big - splited[-1]
    splited.append(big)
    

def _main5():
    imgpath = r'COCO_test2015_000000377928.png'
    img = np.array(Image.open(imgpath))
    split_saliency(img, 3, 6)
    
    
    



if __name__ == '__main__':
    #_main2()
    _main5()
    
    
    

