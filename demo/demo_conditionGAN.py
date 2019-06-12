'''
Created on 2019年3月6日

@author: jinglingzhiyu
'''
import torch
import os, random
import frame_of_wing
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pylab
from frame_of_wing.configs.options import base_parser, train_options_cGAN
from frame_of_wing.my_model.conditionGAN import condition_GAN
from frame_of_wing.my_model.conditionGAN_2 import cGAN
from frame_of_wing.control.control_gan import local_train_for_gan_pytorch

def gen_all_csv(root_path, save_root=None):
#读取指定文件夹下的train,val,test中的全部文件（期望为图片）并返回三个df文件
    train_file = root_path + '//' + 'train'
    val_file   = root_path + '//' + 'val'
    test_file  = root_path + '//' + 'test'
    df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_train['img_path'] = [train_file + '//' + i for i in os.listdir(train_file)]
    df_val['img_path'] = [val_file + '//' + i for i in os.listdir(val_file)]
    df_test['img_path'] = [test_file + '//' + i for i in os.listdir(test_file)]
    if save_root is not None:
        df_train.to_csv(save_root + '//' + 'train.csv', index=False)
        df_val.to_csv(save_root + '//' + 'val.csv', index=False)
        df_test.to_csv(save_root + '//' + 'test.csv', index=False)
    return df_train, df_val, df_test

def gen_split_csv(root_path, save_root=None):
#读取指定文件夹下的trainA,valA,testA,trainB,valB,testB中的全部文件（期望为图片）并返回三个df文件
    train_A, train_B = root_path + '//' + 'trainA', root_path + '//' + 'trainB'
    val_A, val_B     = root_path + '//' + 'valA', root_path + '//' + 'valB'
    test_A, test_B   = root_path + '//' + 'testA', root_path + '//' + 'testB'
    df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_train['img_path'] = [train_A + '//' + i for i in os.listdir(train_A)]
    df_val['img_path']  = [val_A + '//' + i for i in os.listdir(val_A)]
    df_test['img_path'] = [test_A + '//' + i for i in os.listdir(test_A)]
    df_train['label_path'] = [train_B + '//' + i for i in os.listdir(train_B)]
    df_val['label_path']  = [val_B + '//' + i for i in os.listdir(val_B)]
    df_test['label_path'] = [test_B + '//' + i for i in os.listdir(test_B)]
    if save_root is not None:
        df_train.to_csv(save_root + '//' + 'train.csv', index=False)
        df_val.to_csv(save_root + '//' + 'val.csv', index=False)
        df_test.to_csv(save_root + '//' + 'test.csv', index=False)
    return df_train, df_val, df_test

def _main():
    #设置基本信息(部分全局预设置也在optinos中完成)
    options = base_parser()
    args = options.initialize([train_options_cGAN])
    
    #读取数据
    if(args.run_from_python):
        #args.data_root = r'D:\精灵之羽\羽\科大线\190225\others\pytorch-CycleGAN-and-pix2pix-master\datasets\facade_split'
        #args.csv_root = r'D:\精灵之羽\羽\科大线\190225\others\pytorch-CycleGAN-and-pix2pix-master\datasets\facade_split'
        args.data_root = r'F:\data\saliency'
        args.csv_root  = r'F:\data\saliency'
    if(args.need_gen_csv):
        train_data, val_data, test_data = gen_split_csv(args.data_root, save_root=args.csv_root)                    #制作csv文件
    else:
        train_data, val_data, test_data = pd.read_csv(args.csv_root + '//' + 'train.csv'), pd.read_csv(args.csv_root + '//' + 'val.csv'), pd.read_csv(args.csv_root + '//' + 'test.csv')
    
    #建立模型
    from frame_of_wing.my_model.VAE_cGAN import condition_vae_GAN
    model = condition_GAN(args)
    #model.load_state_dict(torch.load(r'E:\workspace\frame_of_wing\demo\temp_model\lowest_loss.pth.tar')['state_dict'])   #加载模型
    
    #开始训练
    train_cGAN = local_train_for_gan_pytorch(model, args)
    train_cGAN.train(train_data, val_data)                               #训练+验证
    train_cGAN.test(val_data)                                           #测试

    


if __name__ == '__main__':
    _main()

