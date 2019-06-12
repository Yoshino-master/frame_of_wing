'''
Created on 2019年3月7日

@author: jinglingzhiyu
'''
import os, time, copy, pickle
import pandas as pd
import torch.optim as optim
import torch
import torchvision
import numpy as np
import functools
from torch.utils.data import DataLoader
from matplotlib import pylab
from PIL import Image
from frame_of_wing.preprocess.pre_GAN import MyDataset_cGAN, transform_cGAN_train_data, transform_cGAN_train_label, transform_cGAN_test
from frame_of_wing.other_utils.recorder import recorders_one_epoch, AverageMeter, tensorboard_writer, method_AverageMeter, get_recorders
from frame_of_wing.other_utils.saver import saver_general, saver_imgs
from frame_of_wing.other_utils.log import logger_general
from frame_of_wing.my_model.conditionGAN import condition_GAN
from frame_of_wing.display.html import ToHtmlFile
from frame_of_wing.postprocess.gathor import my_torch2imgs, torch2np, estimator, general_trans
from frame_of_wing.evaluation.double_img import CC, MAE


def save_result_func(img, path, epoch):
    if os.path.exists(os.path.join(path, str(epoch))) is False:
        os.mkdir(os.path.join(path, str(epoch)))
    for i in range(img.shape[0]):
        t_img = img[i]
        t_img = torch2np(t_img)
        t_img = (((t_img + 1.0) / 2.0) * 255).astype('uint8')
        t_img = Image.fromarray(t_img)
        t_img.save(path + '//' + str(epoch) + '//' + str(i) + '.png')

def torch2npimg_reshape(images, hlist, wlist):
    #将tensor转换为numpy并reshape到原始图片大小
    trans = torchvision.transforms.ToPILImage()
    imgset = []
    for i in range(images.shape[0]):
        img = (images[i] + 1.0) / 2.0
        img = trans(img).resize((hlist[i], wlist[i]))
        imgset.append(np.array(img))
    return imgset

def test_epoch(test_loader, model, options, mode='return_all_data'):
    model.set_eval()
    #设置记录
    recorders = recorders_one_epoch(['loss_g_gan', 'loss_g_l', 'loss_d'], record_all=True)
    recorders['cc'] = method_AverageMeter(CC, record_all=True)                                 #设置CC指标估计器
    recorders['mae']= method_AverageMeter(MAE, record_all=True)                                #设置MAE指标估计器
    write_recorders = {'loss_d' : recorders['loss_d'], 'loss_g_gan' : recorders['loss_g_gan'], 'loss_g_l' : recorders['loss_g_l'],
                       'cc' : recorders['cc'], 'mae' : recorders['mae']}
    tfb_path = os.path.join(options.model_root, 'tf_board')
    counter  = estimator(recorders)                                              #设置估计器(用于计算各种指标)
    test_root = options.model_root + '//' + 'test_results'
    
    imageset, outset, targetset = [], [], []
    for i, (images, target, info) in enumerate(test_loader):
        batch_start = time.time()
        
        #运行模型
        image_var = images.cuda(async=True)
        label = target.cuda(async=True)
        with torch.no_grad():
            gen_out = model(image_var, label)
            loss_g_gan, loss_g_l, loss_d = model.get_loss()
            outset   += torch2npimg_reshape(gen_out.data.cpu(), info['h'], info['w'])
            imageset += torch2npimg_reshape(images.data.cpu(), info['h'], info['w'])
            targetset+= torch2npimg_reshape(target.data.cpu(), info['h'], info['w'])
        
        #更新信息
        batch_end = time.time()
        recorders['batch_time'].update(batch_end - batch_start, n=options.batch_size)
        recorders['loss_g_gan'].update(loss_g_gan.item())
        recorders['loss_g_l'].update(loss_g_l.item())
        recorders['loss_d'].update(loss_d.item())
        
        #打印测试信息
        if i % options.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_G_GAN {loss_g_gan.val:.4f} ({loss_g_gan.avg:.4f})\t'
                  'Loss_G_L {loss_g_l.val:.4f} ({loss_g_l.avg:.4f})\t'
                  'Loss_D {loss_d.val:.4f} ({loss_d.avg:.4f})\t'.format(
                i, len(test_loader), batch_time=recorders['batch_time'], loss_g_gan=recorders['loss_g_gan'],
                loss_g_l=recorders['loss_g_l'], loss_d=recorders['loss_d']))
    
    #计算指标
    for i in range(len(outset)):
        counter.update([outset[i], targetset[i]])             #更新指标
    
    if(mode=='return_all_data'):
        return recorders, imageset, outset, targetset
    else:
        return recorders
        
def validate_epoch(val_loader, model, options, save_model=False, need_show=False, **args):
    model.set_eval()
    #基本设置
    recorders = recorders_one_epoch(['loss_g_gan', 'loss_g_l', 'loss_d'])     #设置记录器
    recorders['cc'] = method_AverageMeter(CC)                                 #设置CC指标估计器
    recorders['mae']= method_AverageMeter(MAE)                                #设置MAE指标估计器
    write_recorders = {'loss_d' : recorders['loss_d'], 'loss_g_gan' : recorders['loss_g_gan'], 'loss_g_l' : recorders['loss_g_l'],
                       'cc' : recorders['cc'], 'mae' : recorders['mae']}
    tfb_path = os.path.join(options.model_root, 'tf_board')
    writer = tensorboard_writer(write_recorders, tfb_path, showname='val'+str(args['epoch']))     #设置写入器(用于实时记录并显示训练信息)
    counter = estimator(recorders)                                             #设置估计器(用于计算各种指标)
    transformer = functools.partial(my_torch2imgs, trans=general_trans())      #设置转换器(后处理相关转换)
    val_root  = options.middle_root + '//' + str(args['epoch']) + '//' + 'val_save'
    if os.path.exists(val_root) is False:
        os.mkdir(val_root)
    
    #开始训练
    for i, (images, target, info) in enumerate(val_loader):
        batch_start = time.time()
        #运行模型
        image_var = images.cuda(async=True)
        label = target.cuda(async=True)
        with torch.no_grad():
            gen_out = model(image_var, label)
            loss_g_gan, loss_g_l, loss_d = model.get_loss()
        
        #展示结果
        if need_show is True:
            s_label = torch2np(target[0].data.numpy())
            s_label = (s_label + 1.0) / 2.0
            s_sour  = torch2np(image_var[0].cpu().data.numpy())
            s_sour  = (s_sour + 1.0) / 2.0
            img = (gen_out.cpu().data.numpy() + 1.0) / 2.0
            img = torch2np(img[0])
            fig = pylab.figure()
            ax1 = fig.add_subplot(131)
            ax1.imshow(s_sour)
            ax2 = fig.add_subplot(132)
            ax2.imshow(img)
            ax3 = fig.add_subplot(133)
            ax3.imshow(s_label)
            pylab.show()
            need_show = False
        
        #更新验证信息
        batch_end = time.time()
        recorders['batch_time'].update(batch_end - batch_start, n=options.batch_size)
        recorders['loss_g_gan'].update(loss_g_gan.item())
        recorders['loss_g_l'].update(loss_g_l.item())
        recorders['loss_d'].update(loss_d.item())
        
        #打印验证信息
        if i % options.print_freq == 0:
            print('TrainVal: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_G_GAN {loss_g_gan.val:.4f} ({loss_g_gan.avg:.4f})\t'
                  'Loss_G_L {loss_g_l.val:.4f} ({loss_g_l.avg:.4f})\t'
                  'Loss_D {loss_d.val:.4f} ({loss_d.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=recorders['batch_time'], loss_g_gan=recorders['loss_g_gan'],
                loss_g_l=recorders['loss_g_l'], loss_d=recorders['loss_d']))
        
        #计算指标及保存batch结果
        images_i, targets_i, outputs_i = transformer(images), transformer(target), transformer(model.output.cpu().data)      #转换图片
        for j in range(len(outputs_i)):
            counter.update([outputs_i[j], targets_i[j]])             #更新指标
        writer.update(i)
        savelist = [os.path.basename(path).replace('.jpg', '.png') for path in info['img_path']]
        saver_imgs(val_root, {'output' : outputs_i}, savelist, imgtype=None, fromnp=True)     #保存图片
    
    #保存epoch记录信息
    with open(os.path.join(val_root, 'recorder.pkl'), 'wb') as f:
        pickle.dump(recorders, f)
    if save_model is True:
        state = {'epoch' : args['epoch'], 'recorders' : recorders,
                 'state_dict' : model.state_dict(), 'options' : options}
        torch.save(state, os.path.join(val_root, 'model_'+str(args['epoch'])+'pth.tar'))
    
    return recorders['loss_g_gan'].get_avg(), recorders['loss_g_l'].get_avg(), recorders['loss_d'].get_avg()
    

def train_epoch(train_loader, model, options, save_model=False, **args):
    model.set_train()
    #设置记录
    recorders = recorders_one_epoch(['loss_d', 'loss_g_gan', 'loss_g_l'], record_all=True)  #设置记录器
    recorders['cc'] = method_AverageMeter(CC, record_all=True)                              #设置CC指标估计器
    recorders['mae']= method_AverageMeter(MAE, record_all=True)                             #设置MAE指标估计器
    write_recorders = {'loss_d' : recorders['loss_d'], 'loss_g_gan' : recorders['loss_g_gan'], 'loss_g_l' : recorders['loss_g_l'],
                       'cc' : recorders['cc'], 'mae' : recorders['mae']}
    tfb_path = os.path.join(options.model_root, 'tf_board')
    writer = tensorboard_writer(write_recorders, tfb_path, showname='train'+str(args['epoch']))     #设置写入器(用于实时记录并显示训练信息)
    counter = estimator(recorders)                                             #设置估计器(用于计算各种指标)
    transformer = functools.partial(my_torch2imgs, trans=general_trans())      #设置转换器(后处理相关转换)
    train_root  = options.middle_root + '//' + str(args['epoch']) + '//' + 'train_save'
    if os.path.exists(train_root) is False:
        os.mkdir(train_root)
    
    #开始训练
    for i, (images, target, info) in enumerate(train_loader):
        batch_start = time.time()
        #训练并计算损失
        image_var = images.cuda()
        label = target.cuda()
        model.optimize(image_var, label)
        loss_g_gan, loss_g_l, loss_d = model.get_loss()
        
        #更新训练信息
        batch_end = time.time()
        recorders['batch_time'].update(batch_end - batch_start, n=options.batch_size)
        recorders['loss_d'].update(loss_d.item())
        recorders['loss_g_gan'].update(loss_g_gan.item())
        recorders['loss_g_l'].update(loss_g_l.item())
        
        #打印训练信息
        if((i%options.print_freq) == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_D {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
                  'Loss_G_GAN {loss_g_gan.val:.4f} ({loss_g_gan.avg:.4f})\t'
                  'Loss_G_L {loss_g_l.val:.4f} ({loss_g_l.avg:.4f})\t'.format(
                args['epoch'], i, len(train_loader), batch_time=recorders['batch_time'],
                loss_d=recorders['loss_d'], loss_g_gan=recorders['loss_g_gan'], loss_g_l=recorders['loss_g_l']))
        
        #计算指标及保存batch结果
        images_i, targets_i, outputs_i = transformer(images), transformer(target), transformer(model.output.cpu().data)
        for j in range(len(outputs_i)):
            counter.update([outputs_i[j], targets_i[j]])             #更新指标
        writer.update(i)
#         savelist = [os.path.basename(path).replace('.jpg', '.png') for path in info['img_path']]
#         saver_imgs(train_root, {'output' : outputs_i}, savelist, imgtype=None, fromnp=True)     #保存图片
    
    #保存epoch记录信息及模型
    with open(os.path.join(train_root, 'recorder.pkl'), 'wb') as f:
        pickle.dump(recorders, f)
    if save_model is True:
        state = {'epoch' : args['epoch'], 'recorders' : recorders,
                 'state_dict' : model.state_dict(), 'options' : options}
        torch.save(state, os.path.join(train_root, 'model_'+str(args['epoch'])+'pth.tar'))
    return recorders['loss_g_gan'].get_avg(), recorders['loss_g_l'].get_avg(), recorders['loss_d'].get_avg()
        

class local_train_for_gan_pytorch():
    def __init__(self, model, options):
        self.model = model
        self.options = options
        if os.path.exists(self.options.model_root) is False:
            os.mkdir(self.options.model_root)
    
    def train(self, train_data, val_data, turn_best_model=True):
        print('start training')
        #设置预处理方式函数
        train_data = MyDataset_cGAN(train_data, transform_data=transform_cGAN_train_data, transform_label=transform_cGAN_train_label)
        val_data   = MyDataset_cGAN(val_data, transform_data=transform_cGAN_train_data, transform_label=transform_cGAN_train_label, need_show=False)
        train_loader = DataLoader(train_data, batch_size=self.options.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_data, batch_size=self.options.batch_size, shuffle=True,  num_workers=0)
        
        #设置记录及相关初始化参数
        recorders_train = get_recorders(['loss_train_g_gan', 'loss_train_g_l', 'loss_train_d'], record_all=True)
        recorders_val   = get_recorders(['loss_val_g_gan', 'loss_val_g_l', 'loss_val_d'], record_all=True)
        recorders = dict(recorders_train, **recorders_val)
        tfb_path = os.path.join(self.options.model_root, 'tf_board')
        writer = tensorboard_writer(recorders, tfb_path, showname='train_val')
        logger = logger_general(self.options.model_root)
        lowest_loss_val = 10000.0
        lowest_loss_train = 10000.0
        self.options.middle_root = self.options.model_root + '//' + 'middle_result'
        if os.path.exists(self.options.middle_root) is False:
            os.mkdir(self.options.middle_root)
        
        if self.options.from_checkpoint is not None:                       #如果设置从断点开始运行,则加载之前的模型及训练参数
            logger.write_line('restart from checkpoint')
            save_file = torch.load(self.options.from_checkpoint + '//' + 'current.pth.tar')
            self.model.load_state_dict(save_file['state_dict'])
            self.options = save_file['options']
            self.options.start_epoch = save_file['epoch']
            optimizer = optim.Adam(self.model.parameters(), self.options.lr, betas=(0.5, 0.999), amsgrad=True)
            lowest_loss_val     = save_file['lowest_loss_val']
            lowest_loss_train   = save_file['lowest_loss_train']
            is_lowest_loss_val  = save_file['is_lowest_loss_val']
            is_lowest_loss_train= save_file['is_lowest_loss_train']
        
        if self.options.evaluate:
            validate_epoch(val_loader, self.model)
        else:
            for epoch in range(self.options.start_epoch, np.sum(self.options.stage_epochs)):
                if os.path.exists(os.path.join(self.options.middle_root, str(epoch))) is False:
                    os.mkdir(os.path.join(self.options.middle_root, str(epoch)))
                #训练及验证模型
                temp_loss_g_gan, temp_loss_g_l, temp_loss_d = train_epoch(train_loader, self.model, self.options, save_model=True,epoch=epoch)
                recorders_train['loss_train_g_gan'].update(temp_loss_g_gan)
                recorders_train['loss_train_g_l'].update(temp_loss_g_l)
                recorders_train['loss_train_d'].update(temp_loss_d)
                temp_loss_g_gan, temp_loss_g_l, temp_loss_d   = validate_epoch(val_loader, self.model, self.options, True, epoch=epoch)
                recorders_val['loss_val_g_gan'].update(temp_loss_g_gan)
                recorders_val['loss_val_g_l'].update(temp_loss_g_l)
                recorders_val['loss_val_d'].update(temp_loss_d)
                writer.update(epoch)
                
                #模型保存
                is_lowest_loss_val    = recorders_val['loss_val_g_l'].val  < lowest_loss_val
                lowest_loss_val = min(recorders_val['loss_val_g_l'].val, lowest_loss_val)
                is_lowest_loss_train  = recorders_train['loss_train_g_l'].val < lowest_loss_train
                lowest_loss_train_val = min(recorders_train['loss_train_g_l'].val, lowest_loss_train)
                #recorders = [loss_train_g_gan, loss_train_g_l, loss_train_d, loss_val_g_gan, loss_val_g_l, loss_val_d]
                state = {'is_lowest_loss_val'  : is_lowest_loss_val,
                         'is_lowest_loss_train': is_lowest_loss_train,
                         'lowest_loss_val'  : lowest_loss_val,
                         'lowest_loss_train': lowest_loss_train,
                         'epoch'          : epoch,
                         'state_dict'     : self.model.state_dict(),
                         'lr'             : self.options.lr,
                         'options'        : self.options,
                         'recorders'      : recorders}
                saver_general(self.options.model_root, False, is_lowest_loss_val, state)        #保存模型
                log_info = {'epoch'       : epoch,
                            'lowest_loss_val'   : lowest_loss_val,
                            'current_loss_val'  : temp_loss_g_l,
                            'lowest_loss_train' : lowest_loss_train,
                            'current_loss_train': temp_loss_g_l}
                logger.write(log_info)                                                      #写入日志
                
                #调整训练策略
                self.model.adjust_stratage_stage(epoch)
                        
        if turn_best_model is True:
            self.model.load_state_dict(torch.load(self.options.model_root + '//' + 'lowest_loss.pth.tar')['state_dict'])
        
    def test(self, test_data):
        print('start testing')
        #设置预处理方式函数
        test_data     = MyDataset_cGAN(test_data, transform_data=transform_cGAN_test, transform_label=transform_cGAN_test)
        test_loader   = DataLoader(test_data, batch_size=self.options.batch_size, shuffle=False,  num_workers=0)
        
        #测试模型
        recorders, imageset, outset, targetset = test_epoch(test_loader, self.model, self.options)
        print('the result of value-cc is:' + str(recorders['cc'].get_avg()))
        print('the result of value-mae is:' + str(recorders['mae'].get_avg()))
        
        #保存结果图片
        data_dict = {'image' : imageset,
                     'output': outset,
                     'label' : targetset}
        save_root = self.options.model_root + '//' + 'save_result'
        filetrack = saver_imgs(save_root, data_dict, fromnp=True)
        state = {'recorders' : recorders, 'options' : self.options}
        with open(os.path.join(save_root, 'test_result.pkl'), 'wb') as f:
            pickle.dump(state, f)
        
        #将结果写入html文件
        ToHtmlFile(os.path.join(self.options.model_root, 'index.html'), filetrack)
        
        



