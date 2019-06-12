'''
Created on 2019年6月3日

@author: jinglingzhiyu
'''
import struct
import numpy as np

def read_imgraw_gray(path):
#读取raw图片文件函数,默认height和width使用int32格式,图片数据使用单byte存储
#返回:img:图片数据
#    :h,w为图片高和宽
#暂时只支持处理处理灰度图像
    with open(path, 'rb') as f:
        context = f.read()
    h, w = struct.unpack('I', context[:4])[0], struct.unpack('I', context[4:8])[0]
    img = [int(i) for i in context[8:]]
    img = np.array(img).reshape((w,h)).astype('uint8')
    return img, (h,w)

def save_imgraw_gray(img, path):
#存储raw图片文件,默认height和width使用int32格式,图片数据使用单byte存储
#暂时只支持处理处理灰度图像
    img = img.astype('uint8')
    with open(path, 'wb') as f:
        img_shape = np.array(img.shape).astype('int32')
        f.write(img_shape[1])
        f.write(img_shape[0])
        f.write(img)



