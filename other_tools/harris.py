'''
Created on 2019年6月10日

@author: jinglingzhiyu
'''
import numpy as np
from scipy.ndimage import filters
from matplotlib import pylab
from PIL import Image


def compute_harris_response(img, sigma=3):
#计算Harris特征矩阵
    imx = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma,sigma), (0,1), imx)
    imy = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma,sigma), (1,0), imy)
    #计算Harris矩阵的分量
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    #计算特征值和迹
    Wdet = Wxx*Wyy - Wxy**2
    Wtr  = Wxx + Wyy
    return Wdet / Wtr

def get_harris_points(harrisim, min_dist=10, threshold=0.1):
#使用非极大值抑制提取角点
    #寻找高于阈值的候选角点
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    #得到候选点的坐标及响应值
    coords = np.array(harrisim_t.nonzero()).T
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    #对候选点按照Harris响应值进行排序
    index = np.argsort(candidate_values)
    #将可行点的位置保存到数组中
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    return filtered_coords

def plot_harris_points(image, filtered_coords):
#绘制Harris检测结果
    pylab.figure()
    pylab.gray()
    pylab.imshow(image)
    pylab.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    pylab.axis('off')
    pylab.show()

def get_descriptors(image, filtered_coords, wid=5):
#获取给定点集附近[-wid:wid,-wid:wid]大小的区域作为描述子
    desc = []
    for i in range(len(filtered_coords)):
        patch = image[(filtered_coords[i][0]-wid):(filtered_coords[i][0]+wid),
                      (filtered_coords[i][1]-wid):(filtered_coords[i][1]+wid)].flatten()
        desc.append(patch)
    return desc

def desc_match(desc1, desc2, threshold=0.5):
#从desc2中选择出与desc1中点集匹配的点集
    n = len(desc1[0])
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if(ncc_value > threshold):
                d[i,j] = ncc_value
    ndx = np.argsort(-d)
    matchsores = ndx[:,0]
    return matchsores
    
def match_twosided(desc1, desc2, threshold=0.5):
#双向匹配点集
    matches_12 = desc_match(desc1, desc2, threshold)
    matches_21 = desc_match(desc2, desc1, threshold)
    for i in range(matches_12.shape[0]):
        if(matches_12[i]<0):
            print('怎么回事?')
            raise
    for i in range(matches_12.shape[0]):
        if(matches_21[matches_12[i]] != i):
            matches_12[i] = -1
    return matches_12

def append_images(img1, img2):
    rows1 = img1.shape[0]
    rows2 = img2.shape[0]
    if(rows1 < rows2):
        img1 = np.concatenate((img1, np.zeros((rows2-rows1, img1.shape[1]))), axis=0)
    elif(rows2 < rows1):
        img2 = np.concatenate((img2, np.zeros((rows1-rows2, img2.shape[1]))), axis=0)
    return np.concatenate((img1, img2), axis=1)

def plot_matches(img1, img2, locs1, locs2, matchscores, show_below=True):
    pylab.figure()
    pylab.gray()
    img3 = append_images(img1, img2)
    if show_below:
        img3 = np.vstack((img3, img3))
    pylab.imshow(img3)
    clos1 = img1.shape[1]
    count =0
    for i, m in enumerate(matchscores):
        if(m > 0):
            pylab.plot([locs1[i][1], locs2[m][1]+clos1], [locs1[i][0], locs2[m][0]], 'c')
            if(count > 10):
                break
            count += 1
    pylab.axis('off')
    pylab.show()

if __name__ == '__main__':
#以下为使用harris算法对两张图的特征点进行匹配的示意程序
    root1 = r'E:\workspace\scan_region_path_explore\model_v1\match1.jpg'
    img1 = np.array(Image.open(root1).convert('L'))
    res1 = compute_harris_response(img1)
    poi1 = get_harris_points(res1)
    des1 = get_descriptors(img1, poi1)
    root2 = r'E:\workspace\scan_region_path_explore\model_v1\match2.jpg'
    img2 = np.array(Image.open(root2).convert('L'))
    res2 = compute_harris_response(img2)
    poi2 = get_harris_points(res2)
    des2 = get_descriptors(img2, poi2)
    matches = match_twosided(des1, des2)
    plot_matches(img1, img2, poi1, poi2, matches)
    


