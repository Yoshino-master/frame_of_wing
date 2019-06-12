'''
Created on 2019年3月25日

@author: jinglingzhiyu
'''
import time, copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab

def im2double(im):
    return cv2.normalize(im.astype('float'),
                         None,
                         0.0, 1.0,
                         cv2.NORM_MINMAX)

def CC(imga, imgb, EPS = 1e-12):
#函数功能:计算两张图片的相关系数
    imga = imga.reshape(1,-1)
    imgb = imgb.reshape(1,-1)
    imgs = np.concatenate((imga, imgb), axis=0)
    cc = np.cov(imgs)
    return (cc[0,1] + EPS) / (np.sqrt(cc[0,0]) * np.sqrt(cc[1,1]) + EPS)

def CCs(imgas, imgbs, EPS=1e-12):
#函数功能:计算两个图片集的相关系数
    if(len(imgas) != len(imgbs)):
        raise Exception('the len of imags must be the same as imgs')
    results = []
    for i in range(len(imgas)):
        imga, imgb = imgas[i], imgbs[i]
        imga = imga.reshape(1,-1)
        imgb = imgb.reshape(1,-1)
        imgs = np.concatenate((imga, imgb), axis=0)
        cc = np.cov(imgs)
        results.append((cc[0,1] + EPS) / (np.sqrt(cc[0,0]) * np.sqrt(cc[1,1]) + EPS))
    return results

def MAE(imga, imgb, EPS=1e-16):
#函数功能:计算两张图片的绝对平均误差
    imga = (imga + EPS) / (np.max(imga) + EPS)
    imgb = (imgb + EPS) / (np.max(imgb) + EPS)
    res = np.abs(imga - imgb)
    return np.mean(res)

def MAEs(imgas, imgbs, EPS=1e-16):
#函数功能:计算两个图片集的绝对平均误差
    if(len(imgas) != len(imgbs)):
        raise Exception('the len of imags must be the same as imgs')
    results = []
    for i in range(len(imgas)):
        imga, imgb = imgas[i], imgbs[i]
        imga = (imga + EPS) / (np.max(imga) + EPS)
        imgb = (imgb + EPS) / (np.max(imgb) + EPS)
        res = np.abs(imga - imgb)
        results.append(np.mean(res))
    return results

def PR(imga, imgb, show=False, EPS=2.2204e-16):
#函数功能:得到PR曲线
#注:这里imga代表预测样例,imgb代表真实样例
    if(imga.shape != imgb.shape):
        raise Exception('the shapes of imga and imgb must be the same')
    precises, recalls = [], []
    _,y = cv2.threshold(imgb, 127, 1, cv2.THRESH_BINARY)
    for i in range(256):
        x = copy.deepcopy(imga)
        i = i-1
        _,x = cv2.threshold(x, i, 1, cv2.THRESH_BINARY)
        #allsize = x.shape[0] * x.shape[1]
        small = cv2.bitwise_and(x, x, mask=y)
        #big   = cv2.bitwise_or(x, y)
        tp = np.sum(small)
        fp = np.sum(x) - tp
        #tn = allsize - np.sum(big)
        fn = np.sum(y) - tp
        precise = (tp + EPS) / (tp + fp + EPS)
        recall  = (tp + EPS) / (tp + fn + EPS)
        precises.append(precise)
        recalls.append(recall)
    if show is True:
        plt.scatter(recalls, precises)
        plt.title('PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precise')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()
    return precises, recalls

def Fmeasure(imga, imgb, beta2=0.3):
#函数功能:计算单对图片的Fmeasure值
    precises, recalls = PR(imga, imgb, show=False)
    fbs = []
    for i in range(len(precises)):
        fb = (1 + beta2)*precises[i]*recalls[i] / (recalls[i] + beta2 * precises[i])
        fbs.append(fb)
    return np.max(fbs)

def Fmeasures(imgas, imgbs, beta2=0.3):
#函数功能:计算两个图片集的Fmeasure的值
    if(len(imgas) != len(imgbs)):
        raise Exception('the len of imags must be the same as imgs')
    starttime = time.time()
    prec, rec = [], []
    for i in range(len(imgas)):
        precises, recalls = PR(imgas[i], imgbs[i], show=False)
        prec.append(precises)
        rec.append(recalls)
    prec = np.array(prec)
    rec  = np.array(rec)
    prec = np.mean(prec, axis=0)
    rec  = np.mean(rec, axis=0)
    fb = (1 + beta2)*prec*rec / (rec + beta2 * prec)
    endtime = time.time()
    print(endtime - starttime)
    return np.max(fb)






