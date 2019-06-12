'''
Created on 2019年2月1日

@author: jinglingzhiyu
'''
import pandas as pd
import os

cf_protocol_demo = {'name':'classification',
                    'save_path':'label.csv',
                    'root_path':'H://临时拷贝//alitianchi_competition//数据//guangdong_round1_train2_20180916',
                    'details':[{'father':['无瑕疵样本'],'type':['.jpg'],'contain':[]},
                               {'father':['不导电'],'type':['.jpg'],'contain':[]},
                               {'father':['擦花'],'type':['.jpg'],'contain':[]},
                               {'father':['横条压凹'],'type':['.jpg'],'contain':[]},
                               {'father':['桔皮'],'type':['.jpg'],'contain':[]},
                               {'father':['漏底'],'type':['.jpg'],'contain':[]},
                               {'father':['碰伤'],'type':['.jpg'],'contain':[]},
                               {'father':['起坑'],'type':['.jpg'],'contain':[]},
                               {'father':['凸粉'],'type':['.jpg'],'contain':[]},
                               {'father':['涂层开裂'],'type':['.jpg'],'contain':[]},
                               {'father':['脏点'],'type':['.jpg'],'contain':[]},
                               {'father':['伤口','划伤','变形','喷流','喷涂碰伤',
                                          '打白点','打磨印','拖烂','杂色','气泡',
                                          '油印','油渣','漆泡','火山口','碰凹',
                                          '粘接','纹粗','角位漏底','返底','铝屑',
                                          '驳口'],'type':['.jpg','.png'],'contain':[]}]}

def make_cf_gc_protocol(save_path,root_path,mode,imgtype = '.jpg',**args):
    protocol = {}
    protocol['save_path'],protocol['root_path'] = save_path,root_path
    protocol['details'],protocol['name']     = [],'classification'
    if 'path_name' in args.keys():
        protocol['img_name'] = args['path_name']
    else:
        protocol['img_name'] = 'img_path'
    if 'class_name' in args.keys():
        protocol['class_name'] = args['class_name']
    else:
        protocol['class_name'] = 'label'
    if(mode==1):
        for xx in range(len(args['describes'])):
            protocol['details'].append({})
            protocol['details'][xx]['father'] = args['describes'][xx]
            protocol['details'][xx]['type']   = [imgtype]
            protocol['details'][xx]['contain']= []
    else:
        print('待施工')                                #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    return protocol


def gen_cf_ass_csv(protocol):
    #函数功能:生成分类任务的csv文件
    #输入:protocol : gen_csv模块协议
    #输出:img_path : 图像文件list
    #     label    :图像类别标签list
    root_path = protocol['root_path']
    details   = protocol['details']
    save_path = protocol['save_path']
    img_path, label = [], []
    match_1,match_2,match_3 = 1,1,1
    for roots,_,files in os.walk(root_path):
        father = os.path.split(roots)[-1]
        for cur_file in files:
            for classes in range(len(details)):
                for fathers in details[classes]['father']:
                    match_1 = 0
                    if(fathers==father):
                        match_1 = 1
                        break
                for types in details[classes]['type']:
                    match_2 = 0
                    if(cur_file[-4:]==types):
                        match_2 = 1
                        break
                for contains in details[classes]['contain']:
                    match_3 = 0
                    if contains in cur_file:
                        match_3 = 1
                        break
                if((match_1)&(match_2)&(match_3)):
                    img_path.append(roots+'//'+cur_file)
                    label.append(classes)
    label_file = pd.DataFrame({protocol['img_name']: img_path, protocol['class_name']: label})
    if save_path is not None:
        label_file.to_csv(save_path, index=False)
    return img_path, label
                
    

if __name__ == '__main__':
    #使用demo协议生成csv文件
#     from gen_csv.gathor import gen_csv_module
#     gen_csvs = gen_csv_module(gen_cf_ass_csv)                    #建立gen_csv模块
#     gen_csvs(cf_protocol_demo)                                   #生成csv文件
    
    #使用函数输入的形式生成csv文件
    from gen_csv.gather import gen_csv_module
    gen_csvs = gen_csv_module(gen_cf_ass_csv)
    protocol = make_cf_gc_protocol(save_path='label.csv',
                                 root_path='H://临时拷贝//alitianchi_competition//数据//guangdong_round1_train2_20180916',
                                 mode = 1,
                                 describes=[['无瑕疵样本'],['不导电'],['擦花'],['横条压凹'],['桔皮'],['漏底'],
                                            ['碰伤'],['起坑'],['凸粉'],['涂层开裂'],['脏点'],['伤口','划伤','变形','喷流','喷涂碰伤',
                                            '打白点','打磨印','拖烂','杂色','气泡','油印','油渣','漆泡','火山口','碰凹','粘接',
                                            '纹粗','角位漏底','返底','铝屑','驳口']])
    gen_csvs(protocol)



