'''
Created on 2019年3月8日

@author: jinglingzhiyu
'''
import matplotlib.pyplot as plt
import matplotlib



def bars(yset, vset, hset, xlabel=None, ylabel=None, title=None, show_range=None):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    x = range(len(yset[0]))
    width = (1.0-0.2) / len(yset)
    rects = []
    for i in range(len(yset)):
        xi = [width*i + j for j in x]
        rects.append(plt.bar(xi, height=yset[i], width=width, alpha=0.8, label=vset[i]))
    if show_range is not None:
        plt.ylim(show_range[0], show_range[1])
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.xticks([i + width/2 for i in x], hset)
    for rect in rects:
        for column in rect:
            height = column.get_height()
            bias = height / 500
            plt.text(column.get_x() + column.get_width() / 2, height + bias, str(height), ha='center', va='bottom')
    plt.show()





if __name__ == '__main__':

#     dataset = [[0.945383, 0.941835, 0.946476], [0.945383, 0.941835, 0.946476]]                  #CC
#     labelset = ['threthold:0', 'threthold:0.05']
#     typeset = ['testset', 'valset', 'trainset']
#     h_range = [0.85, 1.0]
#     y = 'result'
#     title = 'CC'
    
#     dataset = [[0.033034, 0.035861, 0.036358], [0.033034, 0.035861, 0.036358]]                   #DIS
#     labelset= ['threthold:0', 'threthold:0.05']
#     typeset = ['testset', 'valset', 'trainset']
#     h_range = [0.0, 0.05]
#     y = 'result'
#     title = 'DIS'
    
    dataset = [[0.149510, 0.104125, 0.118614], [0.024750, 0.036522, 0.021201]]
    labelset= ['threthold:0', 'threthold:0.05']
    typeset = ['testset', 'valset', 'trainset']
    h_range = [0.0, 0.2]
    y = 'result'
    title = 'MAE'
    
    bars(dataset, labelset, typeset, ylabel=y, title=title, show_range=h_range)

