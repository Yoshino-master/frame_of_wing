'''
Created on 2019年2月2日

@author: jinglingzhiyu
'''
import pandas as pd
from sklearn.model_selection import train_test_split


def dec_cf_csv(csv_path,protocol):
    if 'read_csv_func' not in protocol.keys():
        data = pd.read_csv(csv_path)
    else:
        data = protocol['read_csv_func'](csv_path)
    if 'split_func' not in protocol.keys():
        return data
    else:
        return protocol['split_func'](data,protocol)

def base_split_func(data,protocol):
    if(protocol['testdata']):
        return data
    if 'random_seed' not in protocol.keys():
        train_data,val_data = train_test_split(data,test_size=(1.0-protocol['train_ratio']), stratify=data[protocol['class_name']])
        if 'test_ratio' in protocol.keys():
            val_data,test_data = train_test_split(val_data,test_size=(protocol['test_ratio']/(1.0-protocol['train_ratio'])), stratify=val_data[protocol['class_name']])
            return train_data,val_data,test_data
    else:
        train_data,val_data = train_test_split(data,test_size=(1.0-protocol['train_ratio']), random_state=protocol['random_seed'], stratify=data[protocol['class_name']])
        if 'test_ratio' in protocol.keys():
            val_data,test_data = train_test_split(val_data,random_state=protocol['random_seed'], stratify=val_data[protocol['class_name']],
                                                  test_size=(protocol['test_ratio']/(1.0-protocol['train_ratio']-protocol['test_ratio'])))
            return train_data, val_data, test_data
    return train_data,val_data
    
def make_cf_dec_protocol(testdata,train_tatio,test_ratio=None,random_seed=None,split_func=base_split_func,path_name='img_path',class_name='label'):
    protocol = {}
    if(testdata):
        protocol['testdata'] = True
        return protocol
    else:
        protocol['testdata'] = False
    protocol['path_name']   = path_name
    protocol['class_name']  = class_name
    protocol['train_ratio'] = train_tatio
    protocol['split_func']  = split_func
    if test_ratio is not None:
        protocol['test_ratio'] = test_ratio
    if random_seed is not None:
        protocol['random_seed'] = random_seed
    return protocol





if __name__ == '__main__':
    from decoder.gather import decoder_module
    decoders = decoder_module(dec_cf_csv)
    protocol = make_cf_dec_protocol(False,0.8, 0.1)
    csv_path = 'label.csv'
    train_data,val_data,test_data = decoders(csv_path,protocol)
    









