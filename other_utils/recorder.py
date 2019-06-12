'''
Created on 2019年3月8日

@author: jinglingzhiyu
'''
from tensorboardX import SummaryWriter
import os

class AverageMeter(object):
#超级好用的训练记录器!!!
    """Computes and stores the average and current value"""
    def __init__(self, record_all=False):
        self.record_all = record_all
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        if self.record_all is True:
            self.record = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.record_all is True:
            self.record.append(self.val)
    
    def get_avg(self):
        return self.avg

class method_AverageMeter(AverageMeter):
#带方法的AverageMeter类
#注意:给定的method只能有一个返回值,否则报错
    def __init__(self, method, record_all=False):
        super(method_AverageMeter, self).__init__()
        self.method = method
        self.record_all = record_all
        self.reset()
    def _update(self, params, n=1):
        self.update(self.method(*params), n)
    def _updates(self, paramslist):
    #批量updates函数
        for i in range(len(paramslist)):
            self.update(self.method(*paramslist[i]), 1)
    
def recorders_one_epoch(namelist = None, record_all=False):
#获取常用的recorders
    recorders = {}
    recorders['batch_time'] = AverageMeter(record_all=record_all)
    recorders['loss']   = AverageMeter(record_all=record_all)
    if namelist is not None:
        for name in namelist:
            recorders[name] = AverageMeter(record_all=record_all)
    return recorders

def get_recorders(namelist, record_all=False):
#自建recorders
    if namelist is None:
        return None
    recorders = {}
    for name in namelist:
        recorders[name] = AverageMeter(record_all=record_all)
    return recorders

class tensorboard_writer():
#将recorders中的内容记录为tensorboard可以打开的文件
#注意:
#1.运行update前需要先更新recorders中的参数
#2.本类别依赖于AverageMeter或者method_AverageMeter方法,即recorders中的所有键值必须为这两个方法之一
    def __init__(self, recorders, save_root, showname=None):
        if showname is None:
            self.showname = 'result'
        else:
            self.showname = showname
        self.recorders = recorders
        self.save_root = save_root
        if os.path.exists(self.save_root) is False:
            os.mkdir(self.save_root)
        self.writer = SummaryWriter(self.save_root)
    
    def get_vals(self):
        logdict = {}
        for keyname, recorder in self.recorders.items():
            logdict[keyname] = recorder.val
        return logdict
    
    def get_avgs(self):
        logdict = {}
        for keyname, recorder in self.recorders.items():
            logdict[keyname] = recorder.avg
        return logdict
    
    def update(self, n):
        self.writer.add_scalars(self.showname, self.get_vals(), n)
    
    def close(self):
        self.writer.close()
            

if __name__ == '__main__':
    def temp_f(x,y,z):
        return x + y + z
    import functools
    temp = functools.partial(temp_f, z=5)
    recorder = method_AverageMeter(temp)
    result = recorder._update([2,3])
    print(recorder.val)
    
    
    

