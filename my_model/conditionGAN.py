'''
Created on 2019年3月7日

@author: jinglingzhiyu
'''
import torch
import os, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from frame_of_wing.preprocess.net import init_net

cfg_generator = {
    'encoder_3v1' : [3, 64, 'b', 128, 'b', 256, 'b', 512, 'b', 512, 'b', 512, 'b', 512, 'b', 512, 'n'],         #规则:输入特征图数目+(输出特征图数目、是否批标准化)
    'decoder_3v1' : [512,  512, 0.5,
                    1024, 512, 0.5,
                    1024, 512, 0.5,
                    1024, 512, 0.0,
                    1024, 256, 0.0,
                    512,  128, 0.0,
                    256,  64,  0.0,
                    128,  3,   -1],
    'decoder2_3v1': [512,  512, 0.5,
                     1024, 256, 0.5,
                     512,  128, 0.0,
                     256,  64,  0.0,
                     128,  2,   -1],
    'encoder_1v1' : [3, 64, 'b', 128, 'b', 256, 'b', 512, 'b', 512, 'b', 512, 'b', 512, 'b', 512, 'n'],         #规则:输入特征图数目+(输出特征图数目、是否批标准化)
    'decoder_1v1' : [512,  512, 0.5,
                    1024, 512, 0.5,
                    1024, 512, 0.5,
                    1024, 512, 0.0,
                    1024, 256, 0.0,
                    512,  128, 0.0,
                    256,  64,  0.0,
                    128,  1,   -1],
    'encoder_ver2' : [3, 64, 128, 256, 512, 512, 512, 512, 512],
    'encoder_ver3' : [3, 64, 128, 256, 512, 512]
    }               #规则:(输入特征图数目,输出特征图数目,dropout几率(设为0时表示不需要dropout))

cfg_discriminator = {
                    'discriminator_3v1' : [(6, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 1), (512, 1, 1)],
                    'discriminator_1v1' : [(4, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 1), (512, 1, 1)]
                    }

def make_layers_for_cGAN_encoder(cfg, in_channel=3):
#制作cGAN网络encoder的简化代码
    layers = []
    if in_channel is None:
        in_channel = cfg.pop(0)
    while(len(cfg) != 0):
        v = cfg.pop(0)
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif(isinstance(v, int)):
            conv2d = nn.Conv2d(in_channel, v, kernel_size=4, stride=(2,2), padding=1, bias=False)
            s = cfg.pop(0)
            if(s == 'b'):
                layers += [torch.nn.Sequential(conv2d, nn.BatchNorm2d(v, eps=1e-5, momentum=0.1), nn.LeakyReLU(0.2, inplace=True))]
            elif(s == 'n'):
                layers += [torch.nn.Sequential(conv2d, nn.LeakyReLU(0.2, inplace=True))]
            in_channel = v
    return nn.Sequential(*layers)

def make_layers_for_cGAN_encoder2(cfg):
    layers = []
    if(len(cfg) < 2):
        raise Exception('there are something wrong within the cfg')
    in_channel = cfg.pop(0)
    count = 0
    while(len(cfg) != 0):
        v = cfg.pop(0)
        conv2d = nn.Conv2d(in_channel, v, kernel_size=4, stride=(2,2), padding=1, bias=False)
        if(count == 0):
            layers += [torch.nn.Sequential(conv2d)]
        elif(len(cfg) == 0):
            layers += [torch.nn.Sequential(nn.LeakyReLU(0.2, inplace=True), conv2d)]
        else:
            layers += [torch.nn.Sequential(nn.LeakyReLU(0.2, inplace=True), conv2d, nn.BatchNorm2d(v, eps=1e-5, momentum=0.1))]
        in_channel = v
        count += 1
    return nn.Sequential(*layers)

def make_layers_for_cGAN_decoder2(cfg):
    layers = []
    while(len(cfg) != 0):
        in_c, out_c, dropout_p = cfg.pop(0), cfg.pop(0), cfg.pop(0)
        if(len(cfg) != 0):
            trans_conv2d = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=(2,2), padding=1, bias=False)
        else:
            trans_conv2d = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=(2,2), padding=1)
        if(dropout_p > 0):
            layers += [nn.Sequential(nn.ReLU(inplace=False), trans_conv2d, nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1), nn.Dropout(p=1-dropout_p, inplace=True))]
        elif(dropout_p == 0):
            layers += [nn.Sequential(nn.ReLU(inplace=False), trans_conv2d, nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1))]
        else:
            layers += [nn.Sequential(nn.ReLU(inplace=False), trans_conv2d, nn.Tanh())]
    return nn.Sequential(*layers)

def make_layers_for_cGAN_decoder(cfg):
#制作cGAN网络decoder的简化代码
    layers = []
    while(len(cfg) != 0):
        in_c, out_c, dropout_p = cfg.pop(0), cfg.pop(0), cfg.pop(0)
        if(len(cfg) != 0):
            trans_conv2d = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=(2,2), padding=1, bias=False)
        else:
            trans_conv2d = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=(2,2), padding=1)
        #if(dropout_p > 0):
        if(False):
            layers += [nn.Sequential(nn.Sequential(trans_conv2d, nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1), nn.Dropout(p=1-dropout_p, inplace=True)), nn.ReLU(inplace=True))]
        #elif(dropout_p==0):
        elif(dropout_p >=0):
            layers += [nn.Sequential(nn.Sequential(trans_conv2d, nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1)), nn.ReLU(inplace=True))]
        else:
            layers += [nn.Sequential(nn.Sequential(trans_conv2d), nn.Tanh())]
    return nn.Sequential(*layers)

def make_layers_for_cGAN_discriminator(cfg):
#制作cGAN网络discriminator的简化代码
    layers = []
    for i, (in_c, out_c, stride) in enumerate(cfg):
        if((i == 0) | (i == (len(cfg)-1))):
            bias = True
        else:
            bias = False
        conv2d = nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=bias)
        layers += [conv2d, nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.1), nn.LeakyReLU(0.2, inplace=True)]
    layers.pop(1)
    layers.pop(-1)
    layers.pop(-1)
    return nn.Sequential(*layers)

class generator_v1(nn.Module):
#自制generator(默认输入:256*256)
    def __init__(self):
        super(generator_v1, self).__init__()
        self.encoder = make_layers_for_cGAN_encoder2(cfg_generator['encoder_ver2'])
        self.decoder = make_layers_for_cGAN_decoder2(cfg_generator['decoder_3v1'])
    
    def forward(self, data):
        leap_data = []                               #encoder部分
        for xx in range(len(self.encoder)):
            data = self.encoder[xx](data)
            leap_data.append(data)
        out = []                                     #decoder部分
        for xx in range(len(self.decoder)):
            data = self.decoder[xx](data)
            if(xx != (len(self.decoder)-1)):
                data = torch.cat((leap_data[-(xx+2)], data), dim=1)
            out.append(data)
        return data

class discriminator(nn.Module):
#自制discriminator
    def __init__(self):
        super(discriminator, self).__init__()
        self.discriminator = make_layers_for_cGAN_discriminator(cfg_discriminator['discriminator_3v1'])
    
    def forward(self, data):
        return self.discriminator(data)
    
class condition_GAN(nn.Module):
#自制conditionGAN网络模型
    def __init__(self, options):
        from frame_of_wing.my_model.conditionGAN_2 import UnetGenerator
        super(condition_GAN, self).__init__()
        self.options = options
        self.generator = generator_v1()                    #建立generator
        self.discriminator = discriminator()               #建立discriminator
        self.init_params(self.options)
        self.set_gpu()
    
    def init_params(self, options):
        self.EPS = 1e-12
        self.gan_weight = 1.0
        self.l1_weight  = 100.0                #woc l1 weight这么高，这个明显是冲着视觉效果(发论文)去的
        self.loss_mode = ['default', 'generator', 'discriminator']
        self = init_net(self)
        self.optimizer_G = optim.Adam(self.generator.parameters(), options.lr, betas=(0.5, 0.999), weight_decay = options.weight_decay)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), options.lr, betas=(0.5, 0.999), weight_decay = options.weight_decay)
    
    def adjust_stratage_stage(self, epoch):
        if (epoch+1) in np.cumsum(self.options.stage_epochs)[:-1]:
            self.options.stage += 1
            self.options.lr = self.options.lr / self.options.lr_decay
        self.load_state_dict(torch.load(self.options.model_root + '//' + 'lowest_loss.pth.tar')['state_dict'])
        self.optimizer_G = optim.Adam(self.generator.parameters(), self.options.lr, weight_decay=self.options.weight_decay, betas=(0.5, 0.999), amsgrad=True)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), self.options.lr, weight_decay=self.options.weight_decay, betas=(0.5, 0.999), amsgrad=True)

    def discriminator_loss(self, predict_real, predict_fake):
        return torch.mean(-torch.log(predict_real + self.EPS) - torch.log(1 - predict_fake + self.EPS))
    
    def set_gpu(self):
    #将全部网络放入gpu中
        self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
        self.generator = torch.nn.DataParallel(self.generator).cuda()
    
    def set_train(self):
    #设置网络为训练模式
        self.discriminator.train()
        self.generator.train()
    
    def set_eval(self):
    #设置网络为测试模式
        self.discriminator.eval()
        self.generator.eval()
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def forward(self, image, label):
        self.image, self.label = image, label
        self.output = self.generator(image)
        return self.output
    
    def backward_D(self):
        fakedata = torch.cat((self.image, self.output), 1)
        realdata = torch.cat((self.image, self.label), 1)
        pred_fake = self.discriminator(fakedata.detach())
        self.loss_D_fake = self.GANLoss(pred_fake, False)
        pred_real = self.discriminator(realdata)
        self.loss_D_real = self.GANLoss(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fakedata = torch.cat((self.image, self.output), 1)
        pred_fake= self.discriminator(fakedata)
        self.loss_G_GAN = self.GANLoss(pred_fake, True)
        self.loss_L = F.l1_loss(self.output, self.label) * 100
        self.loss_G = self.loss_G_GAN + self.loss_L
        self.loss_G.backward()
    
    def GANLoss(self, predict, mode):
        if mode is True:
            target = torch.ones(predict.shape).cuda()
        else:
            target = torch.zeros(predict.shape).cuda()
        return F.binary_cross_entropy_with_logits(predict, target)
    
    def get_loss(self):
    #获得当前的损失函数值(注意:模型至少需要运行一次,否则会报错)
        if self.image is None or self.label is None:
            raise Exception('if you want to get the loss of model, please run model at least one time')
        with torch.no_grad():
            self.output = self.generator(self.image)
            fakedata = torch.cat((self.image, self.output), 1)
            realdata = torch.cat((self.image, self.label), 1)
            pred_fake= self.discriminator(fakedata)
            pred_real = self.discriminator(realdata)
            self.loss_G_GAN = self.GANLoss(pred_fake, True)
            self.loss_L = F.l1_loss(self.output, self.label) * 100
            self.loss_G = self.loss_G_GAN + self.loss_L
            self.loss_D_fake = self.GANLoss(pred_fake, False)
            self.loss_D_real = self.GANLoss(pred_real, True)
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            return self.loss_G_GAN.cpu().data.numpy(), self.loss_L.cpu().data.numpy(), \
                self.loss_D
    
    def optimize(self, image, label):
        #更新D
        image, label = image.cuda(), label.cuda()
        self.forward(image, label)
        self.set_requires_grad(self.discriminator, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        #更新G
        self.set_requires_grad(self.discriminator, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        return self.output
    
        
if __name__ == '__main__':
#     from frame_of_wing.configs.options import base_parser, train_options_cGAN
#     options = base_parser()
#     args = options.initialize([train_options_cGAN])
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2'
#     temp_data = torch.ones(1,3,256,256).cuda()
#     temp_data2= torch.ones(1,3,256,256).cuda()
#     model = condition_GAN(args)
#     print(model.generator)
    
    from frame_of_wing.my_model.conditionGAN_2 import UnetGenerator
    model = UnetGenerator(3,3,8, use_dropout=True)
    print(model)
    
    
    
    
    

