'''
Created on 2019年4月20日

@author: jinglingzhiyu
'''
import numpy as np



def split_saliency(img, n=3, max_n=6):
#函数功能:将我们的显著图拆分为不同离散等级的salient patch标签图
#注意:只适用于numpy格式的输入,且要求输入的格式为uint8格式
    splited = []
    big = (img.astype('int32') * n)
    for i in range(n - 1):
        temp = (img / (2 ** (max_n - i))).astype('int32') * (2 ** (max_n - i))
        splited.append(temp.astype('float32'))
        big = big - splited[-1]
    splited.append(big.astype('float32'))
    return splited

def split_saliency_torch(img, n=6, max_n=6, need_init=True):
#函数功能：将我们的显著图拆分为不同离散等级的salient patch标签图
#注意:只适用于torch格式的输入
    img = (img * 255).int()
    splited = []
    big = (img * n)
    for i in range(n - 1):
        temp = (img / (2 ** (max_n - i))) * (2 ** (max_n - i))
        big = big - temp
        splited.append(temp.float() / 255)
    splited.append(big.float() / 255)
    if need_init is True:
        splited.append(img.float() / 255.0)
    return splited
    





if __name__ == '__main__':
    import torch
    import numpy as np
    from matplotlib import pylab
    from PIL import Image
    from torchvision import transforms
    imgpath = r'1.png'
    img = Image.open(imgpath)
#     trans = transforms.ToTensor()
#     img = trans(img)
#     labels = split_saliency_torch(img)
#     for label in labels:
#         label = label.numpy().squeeze()
#         pylab.imshow(label, cmap='Greys_r')
#         pylab.show()
    img = np.array(img)
    labels = split_saliency(img)
    for label in labels:
        pylab.imshow(label, cmap='Greys_r')
        pylab.show()
    




