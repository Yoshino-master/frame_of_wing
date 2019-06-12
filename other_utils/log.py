'''
Created on 2019年3月8日

@author: jinglingzhiyu
'''
import os
import time

class logger_general():
    def __init__(self, root, logname='log.txt'):
        self.root = root
        self.logname = logname
        if os.path.exists(root) is False:
            os.mkdir(root)
        with open(root + '//' + logname, 'a+') as f:
            now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            f.write('experiment time : ' + now_time)
            f.write('\n')
    
    def write(self, infodict, lines=True):
    #写入日志文件
    #当lines为True时按行写入,否则每个写入元素均换行
        with open(self.root + '//' + self.logname, 'a+') as f:
            for key, value in infodict.items():
                f.write('{0} : {1}'.format(key, value))
                if lines is True:
                    f.write('\t')
                else:
                    f.write('\n')
            f.write('\n')
    
    def write_line(self, sentence):
        with open(self.root + '//' + self.logname, 'a+') as f:
            f.write(sentence)
            f.write('\n')
        
    

if __name__ == '__main__':
    temp_dict = {'epoch' : 4, 'loss' : 0.1, 'lowest_loss' : 0.06}
    logger = logger_general('temp')
    logger.write(temp_dict)
    logger.write(temp_dict)
    
    
    
    

