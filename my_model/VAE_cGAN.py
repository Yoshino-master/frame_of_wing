'''
Created on 2019年4月15日

@author: jinglingzhiyu
'''
import torch
import os, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from frame_of_wing.preprocess.net import init_net

class down_block(nn.Module):
#Unet下采样模块
    def __init__(self, in_nc, out_nc, lelu_p=0.2, bn_eps=1e-5, bn_m=0.1, need_bn=True, need_bias=False):
        super(down_block, self).__init__()
        relu = nn.LeakyReLU(lelu_p, inplace=True)
        conv2d = nn.Conv2d(in_nc, out_nc, kernel_size=4, stride=(2,2), padding=1, bias=need_bias)
        bn   = nn.BatchNorm2d(out_nc, eps=bn_eps, momentum=bn_m)
        if need_bn is True:
            self.net = nn.Sequential(relu, conv2d, bn)
        else:
            self.net = nn.Sequential(relu, conv2d)
        
    def forward(self, data):
        return self.net(data)

class up_block(nn.Module):
#上采样模块
    def __init__(self, in_nc, out_nc, bn_eps=1e-5, bn_m=0.1, need_bias=False, dropout_p=0.0, use_tanh=False, use_relu=True):
        super(up_block, self).__init__()
        relu = nn.ReLU(inplace=True)
        trans_conv2d = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=4, stride=(2,2), padding=1, bias=need_bias)
        bn   = nn.BatchNorm2d(out_nc, eps=bn_eps, momentum=bn_m)
        dropout = nn.Dropout(p=1-dropout_p, inplace=True)
        if use_tanh is True:
            tanh = nn.Tanh()
            self.net = nn.Sequential(relu, trans_conv2d, tanh)
        elif use_relu is False:
            self.net = nn.Sequential(trans_conv2d, bn)
        elif(dropout_p == 0):
            self.net = nn.Sequential(relu, trans_conv2d, bn)
        else:
            self.net = nn.Sequential(relu, trans_conv2d, bn, dropout)
    def forward(self, data):
        return self.net(data)

class encoder(nn.Module):
#编码器制作模块
    def __init__(self, in_nc, nfg, max_n, len_n):
        super(encoder, self).__init__()
        if(len_n <= max_n):
            raise Exception('len_n must be smaller to max_n')
        net = []
        net.append(nn.Conv2d(in_nc, nfg, kernel_size=4, stride=(2,2), padding=1, bias=False))
        for i in range(max_n):
            net.append(down_block(nfg * (2**i), nfg * (2 ** (i+1))))
        for i in range(len_n - max_n -1):
            if(i == (len_n - max_n -2)):
                need_bn = False
            else:
                need_bn = True
            net.append(down_block(nfg * (2 ** max_n), nfg * (2 ** max_n), need_bn=need_bn))
        self.net = nn.Sequential(*net)
    def forward(self, data):
        out = []
        for i in range(len(self.net)):
            data = self.net[i](data)
            out.append(data)
        return out, out[-1]

class vae_encoder(nn.Module):
    def __init__(self, in_nc, nfg, max_n, len_n):
        super(vae_encoder, self).__init__()
        if(len_n <= (max_n + 1)):
            raise Exception('len_n must be smaller to max_n')
        net = []
        net.append(nn.Conv2d(in_nc, nfg, kernel_size=4, stride=(2,2), padding=1, bias=False))
        for i in range(max_n):
            net.append(down_block(nfg * (2**i), nfg * (2 ** (i+1))))
        for i in range(len_n - max_n -2):
            if(i == (len_n - max_n -3)):
                need_bn = False
            else:
                need_bn = True
            net.append(down_block(nfg * (2 ** max_n), nfg * (2 ** max_n), need_bn=need_bn))
        self.net = nn.Sequential(*net)
        self.relu = nn.ReLU(inplace=True)
        self.conv2d_mu = nn.Conv2d(nfg * (2 ** max_n), nfg * (2 ** max_n), kernel_size=4, stride=(2,2), padding=1, bias=True)
        self.conv2d_logvar = nn.Conv2d(nfg * (2 ** max_n), nfg * (2 ** max_n), kernel_size=4, stride=(2,2), padding=1, bias=True)
    def forward(self, data):
        out = []
        for i in range(len(self.net)):
            data = self.net[i](data)
            out.append(data)
        data = self.relu(data)
        mu = self.conv2d_mu(data)
        logvar = self.conv2d_logvar(data)
        return out, mu, logvar

class decoder(nn.Module):
    def __init__(self, out_nc, nfg, max_n, len_n, dropout_n=3, dropout_p=0.5):
        super(decoder, self).__init__()
        if(len_n <= (max_n + 1)):
            raise Exception('len_n must be smaller to max_n')
        net = []
        net.append(up_block(nfg * (2 ** max_n), nfg * (2 ** max_n), dropout_p=0.0, use_relu=False))
        for i in range(len_n - max_n - 2):
            net.append(up_block(nfg * (2 ** (max_n+1)), nfg * (2 ** max_n), dropout_p=dropout_p))
            dropout_n -= 1
            if(dropout_n == 0):
                dropout_p = 0.0
        for i in range(max_n):
            now_n = max_n - i
            net.append(up_block(nfg * (2 ** (now_n + 1)), nfg * (2 ** (now_n -1)), dropout_p=dropout_p))
            dropout_n -= 1
            if(dropout_n == 0):
                dropout_p = 0.0
        net.append(up_block(nfg * 2, out_nc, use_tanh=True))
        self.net = nn.Sequential(*net)
    def forward(self, data):
        pass


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder = encoder(3, 64, 3, 8).cuda()
        self.decoder = decoder(3, 64, 3, 8).cuda()
    
    def forward(self, data):
        leap_data, data = self.encoder(data)
        out = []
        for xx in range(len(self.decoder.net)):
            data = self.decoder.net[xx](data)
            if(xx != (len(self.decoder.net)-1)):
                data = torch.cat((data, leap_data[-(xx+2)]), dim=1)
            out.append(data)
        return data

class VAE_Unet_v1(nn.Module):
    def __init__(self):
        super(VAE_Unet_v1, self).__init__()
        self.encoder = vae_encoder(3, 64, 3, 8)
        self.decoder = decoder(3, 64, 3, 8)
        
    def reparametrize(self,mu,logvar):
        lou = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(lou.size()).normal_().cuda()
        return eps.mul(lou).add_(mu)
    def forward(self, data):
        leap_data, self.mu, self.logvar = self.encoder(data)
        data = self.reparametrize(self.mu, self.logvar)
        out = []
        for xx in range(len(self.decoder.net)):
            data = self.decoder.net[xx](data)
            if(xx != (len(self.decoder.net)-1)):
                data = torch.cat((data, leap_data[-(xx+1)]), dim=1)
            out.append(data)
        return data
    def vae_loss(self):
        KLD_element=self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
        KLDloss=torch.sum(KLD_element).mul_(-0.5) / self.mu.size(0)
        return KLDloss

class condition_vae_GAN(nn.Module):
#自制conditionGAN网络模型
    def __init__(self, options):
        super(condition_vae_GAN, self).__init__()
        self.options = options
        from frame_of_wing.my_model.conditionGAN import discriminator
        self.generator = VAE_Unet_v1()                    #建立generator
        self.discriminator = discriminator()               #建立discriminator
        self.init_params(self.options)
        self.set_gpu()
        print('set up model')
    
    def init_params(self, options):
        self.EPS = 1e-12
        self.gan_weight = 0.0
        self.l1_weight  = 1.0                #woc l1 weight这么高，这个明显是冲着视觉效果(发论文)去的
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
        self.vae_loss = self.generator.module.vae_loss() * 5
        self.loss_G = self.loss_G_GAN + self.loss_L + self.vae_loss
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



if __name__ == '__main__':
    from frame_of_wing.configs.options import base_parser, train_options_cGAN
    options = base_parser()
    args = options.initialize([train_options_cGAN])
    temp_data = torch.ones(1,3,256,256).cuda()
    model = condition_vae_GAN(args)
    start = time.time()
    result = model(temp_data, temp_data)
    end = time.time()
    print(end - start)
    print(result.shape)
    
    












