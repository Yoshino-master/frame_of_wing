'''
Created on 2019年3月12日

@author: jinglingzhiyu
'''
import os



def ToHtmlFile(path, datadict):
    if os.path.exists(path):
        index = open(path, 'a')
    else:
        index = open(path, 'w')
    #写入标题栏
    index.write('<html><body><table><tr>')
    index.write('<th>name</th>')
    len_img = -1
    keys_gathor = []
    for itemname in datadict.keys():
        if((len_img != len(datadict[itemname]))&(len_img !=-1)):
            raise Exception('all elements in datadict should have the same length')
        else:
            len_img = len(datadict[itemname])
        keys_gathor.append(itemname)
        index.write('<th>{}</th>'.format(itemname))
    index.write('</tr>')
    
    #写入图片
    for xx in range(len_img):
        index.write('<tr>')
        index.write("<td>{}</td>".format(xx))
        for key in keys_gathor:
            index.write("<td><img src={}></td>".format(datadict[key][xx]))
        index.write("</tr>")
    
    index.write('</html></body></table>')
    















