'''
Created on 2019年2月2日

@author: jinglingzhiyu
'''
import os
import pandas as pd

def cluster_name(root_path, save_path=None, type=['.jpg'], column_name='img_path'):
    #函数功能:搜集指定根文件夹下的全部指定类型文件,并返回对应的df
    img_path = []
    for roots,_,files in os.walk(root_path):
        for xx in range(len(files)):
            for suffix in type:
                if(suffix in files[xx]):
                    img_path.append(roots+'//'+files[xx])
                    continue
    df = pd.DataFrame({column_name:img_path})
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df

class gen_csv_module():
    #类名称:csv文件生成类
    #类功能:建立数据文件的csv文件,以满足训练/测试模型的需要
    def __init__(self,generator):
        self.generator = generator
    def __call__(self,protocol=None):
        if protocol is not None:
            return self.generator(protocol)
        else:
            return self.generator()

if __name__ == '__main__':
    root_path = 'H://临时拷贝//alitianchi_competition//数据//瑕疵样本'
    save_path = 'test.csv'
    cluster_name(root_path, save_path)
    
