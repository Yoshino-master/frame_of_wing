'''
Created on 2019年6月7日

@author: jinglingzhiyu
'''
import cv2
import numpy as np

def DisplayVideo(VideoPath, iterval=20):
#函数功能:播放指定路径下的video文件
#注意:目前只能播放视频，没有声音
    cap = cv2.VideoCapture(VideoPath)
    ret, frame = cap.read()
    while(ret):
        cv2.imshow('image', frame)
        ret, frame = cap.read()
        k = cv2.waitKey(iterval)
        #q键退出
        if (k & 0xff == ord('q')):
            break

def GetVideoFrame(VideoPath=0, MaxF=-1):
#函数功能:获取VideoPath下的视频文件,并以array的形式返回
#参数:
#    VideoPath : 视频存放地址,默认为0,即获取当前设备上的摄像头的照片
#    MaxF      : 返回从视频开始的前n帧结果
    cap = cv2.VideoCapture(VideoPath)
    count = 0
    frames = []
    while True:
        ret, im = cap.read()
        frames.append(im)
        count += 1
        if((count >= MaxF)&(MaxF!=-1)):
            break
    return np.asarray(frames)

def SaveVideo(frames, path, fps=20.0):
#函数功能:将给定的视频流frame保存到path下,fps为视频帧数
#注意 : 默认为avi格式的视频文件
    ImgSize = frames[0].shape
    iscolor = len(ImgSize) == 3
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    outVideo = cv2.VideoWriter(path, fourcc, fps, (ImgSize[1],ImgSize[0]), iscolor)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        outVideo.write(frame)
    outVideo.release()






