'''
Created on 2019年2月2日

@author: jinglingzhiyu
'''
import pandas as pd
import numpy as np
import os,time
import torch
import torch.nn as nn
from preprocess.gathor import MyDataset_cf,transform_typical_cf,transform_typical_cf_non
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import OrderedDict
from tqdm import tqdm
from frame_of_wing.other_utils.recorder import AverageMeter

# 计算top K准确率
def accuracy(y_pred, y_actual, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    final_acc = 0
    maxk = max(topk)
    # for prob_threshold in np.arange(0, 1, 0.01):
    PRED_COUNT = y_actual.size(0)
    PRED_CORRECT_COUNT = 0
    prob, pred = y_pred.topk(maxk, 1, True, True)
    # prob = np.where(prob > prob_threshold, prob, 0)
    for j in range(pred.size(0)):
        if int(y_actual[j]) == int(pred[j]):
            PRED_CORRECT_COUNT += 1
    if PRED_COUNT == 0:
        final_acc = 0
    else:
        final_acc = PRED_CORRECT_COUNT / PRED_COUNT
    return final_acc * 100, PRED_COUNT

# 训练函数(仅训练1个epoch)
def train_epoch(train_loader, model, criterion, optimizer, epoch, print_freq=1):
    batch_time = AverageMeter()                                  #平均每batch运行模型需要的运行时间
    data_time = AverageMeter()                                   #平均每batch数据预处理所需要的时间
    losses = AverageMeter()                                      #损失函数值
    acc = AverageMeter()                                         #精度

    #切换到训练模式
    model.train()
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        #读入数据
        data_time.update(time.time() - end)
        image_var = torch.tensor(images).cuda(async=True)
        
        label = torch.tensor(target).cuda(async=True)

        #运行模型
        y_pred = model(image_var)
        loss = criterion(y_pred, label)

        #计算精度及损失函数值
        prec, PRED_COUNT = accuracy(y_pred.data, target, topk=(1, 1))
        losses.update(loss.item(), images.size(0))
        acc.update(prec, PRED_COUNT)

        #反向传播及梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #测量运行时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))

# 验证函数
def validate_epoch(val_loader, model, criterion, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    #切换到测试模式
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        image_var = torch.tensor(images).cuda(async=True)
        target = torch.tensor(labels).cuda(async=True)

        #计算模型输出
        with torch.no_grad():
            y_pred = model(image_var)
            loss = criterion(y_pred, target)

        #计算精度
        prec, PRED_COUNT = accuracy(y_pred.data, labels, topk=(1, 1))
        
        losses.update(loss.item(), images.size(0))
        acc.update(prec, PRED_COUNT)

        #计算运行时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('TrainVal: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))

    return acc.avg, losses.avg

# 测试函数
def test_epoch(test_loader, model):
    # switch to evaluate mode
    model.eval()
    pred,filename = [],[]
    for i, (images, filepath) in tqdm(enumerate(test_loader)):
        # bs, ncrops, c, h, w = images.size()
        filepath = [i.split('/')[-1] for i in filepath]
        image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

        with torch.no_grad():
            y_pred = model(image_var)
            # get the index of the max log-probability
            y_pred = torch.argmax(y_pred.cpu(), dim=1)
            pred.extend(y_pred.data.tolist())
            filename.extend(filepath)
    return pred,filename

class local_train_for_cf_pytorch():
    def __init__(self, model):
        self.model = model
    
    def adjust_learning_rate(self, lr, lr_decay, weight_decay):
        lr = lr / lr_decay
        return optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)
    
    def save_model(self, state, save_path, is_best_precision, is_lowest_loss):
        torch.save(state, save_path+'//'+'current.pth.tar')
        if(is_best_precision):
            torch.save(state, save_path+'//'+'best_precision.pth.tar')
        if(is_lowest_loss):
            torch.save(state, save_path+'//'+'lowest_loss.pth.tar')
    
    def fit(self, train_data, val_data, turn_best_model=True):
    #函数功能:局部控制模块(训练分类模型)
    #需求输入:以df形式保存的数据
    #         df包含两个列,分别为:img_path和label
        
        #保存路径
        save_path  = 'save_model'
        model_name = 'temp_model'
        # epoch数量，分stage进行，跑完一个stage后降低学习率进入下一个stage
        stage_epochs = [1,2,2]
        # 小数据集上，batch size不宜过大
        batch_size = 6
        # 初始学习率
        lr = 1e-4
        # 学习率衰减系数 (new_lr = lr / lr_decay)
        lr_decay = 5
        # 正则化系数
        weight_decay = 1e-4
        # 训练及验证时的打印频率，用于观察loss和acc的实时变化
        print_freq = 1
        # 是否只验证，不训练
        evaluate = False
        start_epoch = 0
        best_precision = 0
        total_epochs = sum(stage_epochs)
        lowest_loss = 10000
        stage = 1
        
        # GPU ID
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
        
        #设置损失函数和优化方法
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)
        
        #设置预处理方式函数
        train_data = MyDataset_cf(train_data, transform=transform_typical_cf)
        val_data   = MyDataset_cf(val_data, transform=transform_typical_cf)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=True,  num_workers=0)
        
        #训练主代码
        if evaluate:
            validate_epoch(val_loader, self.model, criterion)
        else:
            for epoch in range(start_epoch, total_epochs):
                #训练并验证模型
                train_epoch(train_loader, self.model, criterion, optimizer, epoch)
                precision, avg_loss = validate_epoch(val_loader, self.model, criterion)
                
                #模型数据记录及保存
                is_best_precision = precision > best_precision
                is_lowest_loss    = avg_loss  < lowest_loss
                best_precision    = max(precision, best_precision)
                lowest_loss       = min(lowest_loss, avg_loss)
                
                state = {'is_best_precision' : is_best_precision,
                         'is_lowest_loss'    : is_lowest_loss,
                         'best_precision'    : best_precision,
                         'lowest_loss'       : lowest_loss,
                         'epoch'             : epoch,
                         'state_dict'        : self.model.state_dict(),
                         'lr'                : lr}
                self.save_model(save_path+'//'+model_name)
                
                #判断是否进入下一个stage
                if (epoch+1) in np.cumsum(stage_epochs)[:-1]:
                    stage += 1
                    optimizer = self.adjust_learning_rate(lr, lr_decay, weight_decay)
                    self.model.load_state_dict(torch.load(save_path+'//'+model_name+'//'+'best_precision.pth.tar')['state_dict'])
                    print('Step into next stage')
        
        if turn_best_model is True:
            self.model.load_state_dict(torch.load(save_path+'//'+model_name+'//'+'best_precision.pth.tar')['state_dict'])
        
                    
    def test(self,test_data):
        #设置基本参数
        batch_size = 6
        
        #设置预处理方式函数
        test_data   = MyDataset_cf(test_data,transform=transform_typical_cf_non,test=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,  num_workers=0)
        
        #使用模型进行预测
        pred, filename = test_epoch(test_loader=test_loader, model=self.model)
        
    
        
        
                    
                
                
                
                
                
                
                
                
            
        
        
        
        
        
        
        
        
        
        
        
        
            
            















