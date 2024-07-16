import os
import torch
from torch import nn
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

def data_input_process(modelConfig):
    # input: dataset in modelConfig
    # output: hs, target in CS
    # cude or cpu
    if modelConfig['cpu-cuda'] & torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    sub_band = str(modelConfig['sub_band'])
    # loading data
    data_path = './data/sub-band-' + sub_band + '/' + modelConfig['dataset'] + '-' + sub_band + '.mat'
    mat = io.loadmat(data_path)
    hs = mat['data'].astype(np.float32)
    gt = mat['map'].astype(np.float32)
    hs = torch.tensor(hs).to(device)
    gt = torch.tensor(gt).to(device)
    hs = hs / 1024
    # get target & background
    spectral_sub0 = segment_targ_bkg(hs, gt)

    return spectral_sub0

# def data_drop(data_in):

def load_test_data(modelConfig):
    sub_band = str(modelConfig['sub_band'])
    data_path = './data/sub-band-' + sub_band + '/' + modelConfig['dataset'] + '-' + sub_band + '.mat'
    mat = io.loadmat(data_path)
    hs = mat['data'].astype(np.float32)
    gt = mat['map'].astype(np.float32)
    hs = torch.tensor(hs)
    gt = torch.tensor(gt)
    hs = hs / 1024
    # to vector
    hs_v = hs.contiguous().view(-1, hs.shape[-1])
    hs_CDF = (hs_v)
    return hs_CDF, gt

def RDP_process(modelConfig):
    # input: prior spectrum random victor [num, band]
    # output: BL

    # loading data
    data_path = './data/prior/' + modelConfig['dataset'] + '-prior.mat'
    mat = io.loadmat(data_path)
    target_prior = mat['prior']

    t_mu = np.mean(target_prior, axis=0)
    t_var = np.var(target_prior, axis=0)
    BL = np.sum(np.power(t_mu, 2) + t_var)/modelConfig['sub_band']

    return torch.tensor(BL)

def RDP_Sample(BL, band_num, device):
    # torch.manual_seed(123)
    sample = torch.randn(1, band_num).to(device)
    sample = sample * torch.sqrt(BL)
    # sample = 2*(sample - min_value) / (max_value - min_value) - 1
    sample = sample / 1024
    # RDP_sample_CDF = CDF(sample)
    RDP_sample_CDF = (sample)

    return RDP_sample_CDF

def segment_targ_bkg(hs, gt):
    # input: hs and gt after Avgpooling
    # output: the vector of target and background in a certain scale

    indices = torch.nonzero(gt == 1)
    target_v = hs[indices[:, 0], indices[:, 1]]

    indices = torch.nonzero(gt == 0)
    bkg_v = hs[indices[:, 0], indices[:, 1]]

    hs_v = hs.contiguous().view(-1, hs.shape[-1])

    return [hs_v, target_v, bkg_v]

def CDF(x):
    # input: spectral victor: [batch, sub_band]
    S = torch.sum(x, dim=1, keepdim=True)
    x = x / S
    # x = torch.softmax(x, dim=1)
    x = torch.cumsum(x, dim=1)

    return x

def save_quant_model2file(quant_model, pth):
    param_info = []
    for name, param in quant_model.named_parameters():
        param_info.append(f"Layer: {name}, Shape: {param.shape}")

    # 将参数信息写入文本文件
    with open(pth, 'w') as file:
        file.write('\n'.join(param_info))

if __name__ == '__main__':
    modelConfig = {
        'state': 'train',           # train or test
        'dataset': 'Sandiego',
        'pooling_ksize': [2, 4],    # kernel size of each Avgpooling
        'sub_band': 90,             # number of band in CS
        'cpu-cuda': 0,              # 0-cpu 1-cuda
    }
    # hs_v, target_v = data_input_process(modelConfig)
    BL = RDP_process(modelConfig)