import os
import torch
from torch import nn
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
    
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

    # normalize
    min_value = hs.min()
    max_value = hs.max()
    hs = hs / 1024

    # get target & background
    spectral_sub0 = segment_targ_bkg(hs, gt)

    return spectral_sub0, [min_value, max_value]

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
