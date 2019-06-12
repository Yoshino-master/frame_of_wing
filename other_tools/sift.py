'''
Created on 2019年6月11日

@author: jinglingzhiyu
'''
from PIL import Image
from matplotlib import pylab
import os, cv2
import numpy as np
from scipy.cluster.vq import *

def process_image(imagename, resultname, midpath='', params='--edge-thresh 10 --peak-thresh 5'):
    if(imagename[-3:] != 'pgm'):
        img = Image.open(imagename).convert('L')
        midname = os.path.join(midpath, 'tmp.pgm')
        img.save(midname)
    cmmd = str('sift ' + midname + ' --output=' + resultname + ' ' + params)
    os.system(cmmd)
    os.remove(midname)
    print('processed ' + imagename + ' to ' + resultname)

def read_features_from_file(filename):
#提取计算的sift算子结果
    f = np.loadtxt(filename)
    if(len(f.shape) == 1):
        f = f.reshape(1,*f.shape)
    return f[:,:4], f[:,4:]

def write_features_to_file(filename, locs, desc):
#保存sift文件
    np.savetxt(filename, np.hstack((locs, desc)))
    
def plot_features(img, locs, circle=False):
#绘制带有特征点的图像
    def draw_circle(c,r):
        t = np.arange(0,1.01,0.01) * 2 * np.pi
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        pylab.plot(x,y,'c',linewidth=1)
    pylab.imshow(img)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2])
    else:
        pylab.plot(locs[:,0], locs[:,1],'ob')
    pylab.axis('off')
    
def match_desc(desc1, desc2, dist_ratio=0.6):
#对于第一幅图像中的每个描述子,选取其在第二幅图像中的匹配
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])
    desc1_size = desc1.shape
    matchscores = np.zeros((desc1_size[0],1), 'int')
    desc2t = desc2.T
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:], desc2t)
        dotprods = 0.9999 * dotprods
        indx = np.argsort(np.arccos(dotprods))
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
    return matchscores

def match_twosided(desc1, desc2, dist_ratio=0.6):
    #双向对称版本的match
    matches_12 = match_desc(desc1, desc2, dist_ratio)
    matches_21 = match_desc(desc2, desc2, dist_ratio)
    ndx_12 = matches_12.nonzero()[0]
    #去除不对称的匹配
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
    return matches_12.squeeze()

class Vocabulary(object):
#对给定的图像兴趣点的特征进行聚类,得到视觉词袋模型
    def __init__(self, name):
        self.name = name
        self.voc, self.idf, self.trainingdata = [], [], []
        self.nbr_words = 0
    def train(self, featurefiles, k=100, subsampling=1):
    #对给定文件下下的所有数据文件进行训练
    #imwords为训练后得到的词频表,x轴为图片索引,y轴为视觉单词索引
    #idf为逆文档频率
        nbr_images = len(featurefiles)
        #从文件中读取特征
        descr = []
        descr.append(read_features_from_file(featurefiles[0])[1])
        descriptors = descr[0]
        for i in range(1, nbr_images):
            descr.append(read_features_from_file(featurefiles[i])[1])
            descriptors = np.vstack((descriptors, descr[i]))
        print('finished loading files')
        #Kmeans聚类(这里最后一个参数为迭代次数)
        self.voc, self.distortion = kmeans(descriptors[::subsampling, :], k, 1)
        print('finished clustering')
        self.nbr_words = self.voc.shape[0]
        #遍历所有图像,并投影到词汇上
        self.imwords = np.zeros((nbr_images, self.nbr_words))
        for i in range(nbr_images):
            self.imwords[i] = self.project(descr[i])
        nbr_occurences = np.sum((self.imwords > 0) * 1, axis=0)
        self.idf = np.log((1.0*nbr_images) / (1.0*nbr_occurences+1))
        self.trainingdata = featurefiles
    def project(self, descriptors):
        imhist = np.zeros((self.nbr_words))
        words, distance = vq(descriptors, self.voc)     #将各特征点向量归类到距离最近的聚类中心
        for w in words:
            imhist[w] += 1
        return imhist
    

if __name__ == '__main__':
#以下为使用sift算法对两张图的特征点进行匹配的示意程序
    from frame_of_wing.other_tools.harris import plot_matches
    imgname1 = r'E:\workspace\scan_region_path_explore\model_v1\match1.jpg'
    imgname2 = r'E:\workspace\scan_region_path_explore\model_v1\match2.jpg'
    midpath = r'E:\workspace\scan_region_path_explore\model_v1'
    resultname1 = r'E:\workspace\scan_region_path_explore\model_v1\res1.sift'
    resultname2 = r'E:\workspace\scan_region_path_explore\model_v1\res2.sift'
    img1 = np.array(Image.open(imgname1).convert('L'))
    img2 = np.array(Image.open(imgname2).convert('L'))
#     process_image(imgname1, resultname1, midpath)
#     process_image(imgname2, resultname2, midpath)
    l1, d1 = read_features_from_file(resultname1)
    l2, d2 = read_features_from_file(resultname2)
    print(l1.shape)
    print(d1.shape)
#     matches = match_twosided(d1, d2)
#     plot_matches(img1, img2, l1, l2, matches)



