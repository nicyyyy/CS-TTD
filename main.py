from train import *
from test import *


if __name__ == '__main__':
    modelConfig = {
        'state':            'test',    # train or test
        'dataset':          'Sandiego', # name of the dataset
        'epoch':            1000,       #
        'alpha':            0.2,        # margin
        'n_heads':          5,          # number of multi-head attention
        'd_ff':             512,        # Dimension of feedforward
        'n_layers':         2,          # number of transformer layer
        'n_clip':           5,          # number of clips
        'sub_band':         90,         # number of band in CS
        'cpu-cuda':         0,          # 0-cpu 1-cuda
    }

    if modelConfig['state'] == 'train':
        train(modelConfig)
    elif modelConfig['state'] == 'test':
        Pred_img, PF, PD, threshold, AUC_list = test(modelConfig)