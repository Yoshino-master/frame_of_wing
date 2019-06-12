'''
Created on 2019年2月2日

@author: jinglingzhiyu
'''
import torch,torchvision
import pandas as pd
import numpy as np
import random,os

def _main():
    from decoder.gather import decoder_module
    from decoder.classification import dec_cf_csv, make_cf_dec_protocol
    from preprocess.gathor import MyDataset_cf
    from control.gathor import local_train_for_cf_pytorch
    from my_model import model_v4
    from gen_csv.gathor import gen_csv_module, cluster_name
    from gen_csv.classification import gen_cf_ass_csv, make_cf_gc_protocol
    
    #设置随机种子
    np.random.seed(500)
    torch.manual_seed(500)
    torch.cuda.manual_seed_all(500)
    random.seed(500)
    
    #生成csv文件
    gen_csvs = gen_csv_module(gen_cf_ass_csv)
    protocol = make_cf_gc_protocol(save_path='label.csv',
                                 root_path='H://临时拷贝//alitianchi_competition//数据//guangdong_round1_train2_20180916',
                                 mode = 1,
                                 describes=[['无瑕疵样本'],['不导电'],['擦花'],['横条压凹'],['桔皮'],['漏底'],
                                            ['碰伤'],['起坑'],['凸粉'],['涂层开裂'],['脏点'],['伤口','划伤','变形','喷流','喷涂碰伤',
                                            '打白点','打磨印','拖烂','杂色','气泡','油印','油渣','漆泡','火山口','碰凹','粘接',
                                            '纹粗','角位漏底','返底','铝屑','驳口']])
    gen_csvs(protocol)                                            #生成训练用csv文件
    cluster_name(root_path='H://临时拷贝//alitianchi_competition//数据//guangdong_round1_train2_20180916',
                 save_path='test.csv')                            #生成测试用csv文件
    
    
    #设置基本参数
    file_name = os.path.basename(__file__).split('.')[0]          #获取当前文件名
    current_path = os.getcwd()
    
    #从csv文件中提取数据信息
    decoders = decoder_module(dec_cf_csv)
    dec_protocol = make_cf_dec_protocol(False,0.88)
    train_csv = 'label.csv'
    train_data,val_data = decoders(train_csv,dec_protocol)
    test_data = pd.read_csv('test.csv')
    
    #选择模型
    model = model_v4.v4(num_classes=12)
    model = torch.nn.DataParallel(model).cuda()
    
    #开始训练(选择训练函数)
    my_train = local_train_for_cf_pytorch(model)
    my_train.fit(train_data, val_data)
    
    #测试结果
    my_train.test(test_data)


if __name__ == '__main__':
    _main()

