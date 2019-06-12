'''
Created on 2019年3月6日

@author: jinglingzhiyu
'''
import argparse
import os, random
import torch
import numpy as np
from frame_of_wing.preprocess.gathor import set_random_seed

def train_options_cGAN(parser):
#cGAN的训练配置参数
    parser.add_argument('-lr', default=2e-4, type=float, help='训练阶段的初始学习率')
    parser.add_argument('-lr_decay', default=1.3, type=int, help='每个stage后lr下降的倍数')
    #parser.add_argument('-epoch', default=20, type=int, help='模型需要运行的epoch数目')
    parser.add_argument('-print_freq', default=5, type=int, help='每多少个epoch需要打印运行信息')
    parser.add_argument('-batch_size', default=17, type=int, help='每次放入模型多少数目的数据')
    parser.add_argument('-epoch_stage', default=None, help='当需要使用stage epoch方式训练时使用,不适用时设置为None')
    parser.add_argument('-weight_decay', default=0.0, help='训练时的正则化系数')
    parser.add_argument('-evaluate', default=False, type=bool, help='是否为测试模式(即只预测,不训练),默认为False')
    parser.add_argument('-stage_epochs', default=[15, 5, 5, 5, 10, 10], help='分几个阶段训练')
    parser.add_argument('-start_epoch', default=0, help='从第几个epoch开始运行,当从头训练时默认为0')
    parser.add_argument('-stage', default=1, help='开始时处于哪个阶段')
    parser.add_argument('-input_shape', default=(256, 256), help='输入模型的图片大小')
    return parser

class base_parser():
    def __init__(self, fast_init=True):
        pass
    
    def initialize(self, extra_funcs, need_config=True):
        parser = argparse.ArgumentParser()
        parser.add_argument('-data_root', default=None, type=str, help='数据集存放的根目录,默认为空')
        parser.add_argument('-model_root', default=r'E:\workspace\saliency_dataset\main\saves\pix2pix_temp', type=str, help='模型及相关记录保存的路径,默认为空')
        parser.add_argument('-csv_root', default=None, type=str, help='csv文件生成或读取的路径')
        parser.add_argument('-need_gen_csv', default=False, type=bool, help='是否需要生成csv文件,默认需要(如果已经生成了就不再不要生成了)')
        parser.add_argument('-name', default='default_experiment', type=str, help='本次实验名称,默认为default_experiment')
        parser.add_argument('-from_checkpoint', default=None, help='是否从断点开始运行,设为None意为从头训练,否则则应设置为模型保存的文件夹')
        parser.add_argument('-environment', default='0,2', help='模型运行的环境,使用gpu时设置为数字,使用cpu时设置为cpu')
        parser.add_argument('-run_from_python', default=True, type=bool, help='是否从python程序运行,若设置为否,则默认为从命令行运行')
        parser.add_argument('-random_seed', default=999, help='设置全局随机数种子,若不为数字则表示不设置随机初始化种子')
        for single_func in extra_funcs:                #添加额外的配置参数
            parser = single_func(parser)
        self.args = parser.parse_args()
        if need_config is True:
            self.config()              #执行一些预先设置的信息
        return self.args
        
    def config(self):
        #配置模型运行环境
        if(self.args.environment != 'cpu'):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.environment
            set_random_seed(self.args.random_seed)
    










if __name__ == '__main__':
    option = base_parser()
    args = option.initialize()
    print(args)

