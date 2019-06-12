'''
Created on 2019年2月9日

@author: jinglingzhiyu
'''
import torch,torchvision
import pandas as pd
import numpy as np
import random,os
from decoder.gather import decoder_module
from decoder.classification import dec_cf_csv,make_cf_dec_protocol
from preprocess.gathor import MyDataset_cf
from control.gathor import local_train_for_cf_pytorch
from my_model import model_v4
from PIL import Image

def init_cf_process(block):
    # 随机种子
    np.random.seed(500)
    torch.manual_seed(500)
    torch.cuda.manual_seed_all(500)
    random.seed(500)
    
    #设置
    block.file_name = os.path.basename(__file__).split('.')[0]          #获取当前文件名
    block.current_path = os.getcwd()
    
    #从csv文件中提取数据信息
    block.decoders = decoder_module(dec_cf_csv)
    block.dec_protocol = make_cf_dec_protocol(False,0.88)               #设置训练集和测试集划分比例为88:12
    block.csv_path = 'label.csv'
    block.locktrain_data,val_data = block.decoders(block.csv_path,block.dec_protocol)
    
    #选择模型
    block.model = model_v4.v4(num_classes=12)
    block.model = torch.nn.DataParallel(block.model).cuda()
    
    #选择训练函数
    block.train_func = local_train_for_cf_pytorch(block.model)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class total_control():
    def __init__(self):
        pass
    
    def init_params(self,init_func):
        init_func(self)
    
    
        
    
        




if __name__ == '__main__':
    session = total_control()
    session.init_params(init_cf_process)








