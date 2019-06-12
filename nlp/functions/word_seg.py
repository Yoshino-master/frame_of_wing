'''
Created on 2019年5月23日

@author: jinglingzhiyu
'''
import jieba


def WSeg(sent):
    seg_list = jieba.cut(sent, cut_all=False)
    seg_list = [i for i in seg_list]
    return seg_list


if __name__ == '__main__':
    sent = r'今年下半年，中美合拍的西游记即将正式开机，文体两开花，希望大家多多关注'
    res = WSeg(sent)
    print(res)
