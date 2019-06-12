'''
Created on 2019年6月11日

@author: jinglingzhiyu
'''
import pydot, os
import numpy as np
from PIL import Image
from matplotlib import pylab

def GetTreeByDict(DataTree, name, g=None, fontname="DengXian"):
#使用Pydot获得关系图
    if g is None:
        g = pydot.Dot(graph_type='graph')
    g.add_node(pydot.Node(name, fontname=fontname))
    for k,v in DataTree.items():
        if isinstance(v, dict):
            g = GetTreeByDict(v, k, g, fontname=fontname)
        else:
            g.add_node(pydot.Node(k, fontname=fontname))
        g.add_edge(pydot.Edge(name, k))
    return g

def PlotTreeByDict(DataTree, savepath, fontname="DengXian"):
#使用Pydot绘制关系图
    g = pydot.Dot(graph_type='graph')
    for k,v in DataTree.items():
        g = GetTreeByDict(v, k, g, fontname=fontname)
    g.write_png(savepath, prog='neato.exe', encoding='utf-8')
    img = np.array(Image.open(savepath))
    os.remove(savepath)
    return img

if __name__ == '__main__':
    data = {'约战':{r'折纸鸢一':{r'白毛':None},r'五河琴里':{r'傲娇':None,r'双马尾':None,r'粉毛':None},r'四糸乃':{r'兔子':None,r'蓝毛':None, r'loli':None}}}
    savedir = r'E:\workspace\frame_of_wing\display\graph.png'
    img = PlotTreeByDict(data, savedir)
    pylab.imshow(img, cmap='Greys_r')
    pylab.show()
    
    
    
