'''
Created on 2019年4月4日

@author: jinglingzhiyu
'''
'''
Created on 2019年3月7日

@author: jinglingzhiyu
'''
import torch
import os, time
import torch.nn as nn
from torchvision.models import vgg
import functools
import torch.nn.functional as F

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.netD = nn.Sequential(
                                        nn.Conv2d(6, 64, (4,4), stride=(2,2), padding=(1,1)),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(64, 128, (4,4), stride=(2,2), padding=(1,1), bias=False),
                                        nn.BatchNorm2d(128, eps=1e-5, momentum=0.1),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(128, 256, (4,4), stride=(2,2), padding=(1,1), bias=False),
                                        nn.BatchNorm2d(256, eps=1e-5, momentum=0.1),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(256, 512, (4,4), stride=(1,1), padding=(1,1), bias=False),
                                        nn.BatchNorm2d(512, eps=1e-5, momentum=0.1),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(512, 1, (4,4), stride=(1,1), padding=(1,1)))
    def forward(self, data):
        out = self.netD(data)
        return out

class cGAN(nn.Module):
    def GANLoss(self, predict, mode):
        if mode is True:
            target = torch.ones(predict.shape).cuda()
        else:
            target = torch.zeros(predict.shape).cuda()
        return F.binary_cross_entropy_with_logits(predict, target)
        
    def __init__(self):
        super(cGAN, self).__init__()
        self.generator = UnetGenerator(3,3,8, use_dropout=True)
        self.discriminator = discriminator()
        self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
        self.generator = torch.nn.DataParallel(self.generator).cuda()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
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
        
    
    
    
        
        

def temp_test():
    model = cGAN()
    model.discriminator = torch.nn.DataParallel(model.discriminator).cuda()
    model.generator = torch.nn.DataParallel(model.generator).cuda()
    data = torch.ones(4,3,256,256).cuda()
    target = torch.ones(4,3,256,256).cuda()
    model.optimize(data, target)
    



if __name__ == '__main__':
    temp_test()



