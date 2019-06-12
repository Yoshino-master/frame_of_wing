'''
Created on 2019年3月11日

@author: jinglingzhiyu
'''
import numpy as np
import torch.optim as optim
import torch

def adjust_stratage_stage(epoch, model, options, optimizer):
    try:
        if options.stage_epochs is None:
            raise
        if options.stage is None:
            raise
    except:
        raise Exception('options must have attributes:stage_epochs and stage')
    if (epoch+1) in np.cumsum(options.stage_epochs)[:-1]:
        options.stage += 1
        options.lr = options.lr / options.lr_decay
        model.load_state_dict(torch.load(options.model_root + '//' + 'lowest_loss.pth.tar')['state_dict'])
        optimizer = optim.Adam(model.parameters(), options.lr, weight_decay=options.weight_decay, betas=(0.5, 0.999), amsgrad=True)
        print('Step into next stage')
    return optimizer













