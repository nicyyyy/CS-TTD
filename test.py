import torch
from utils import *
from model import *
import time
from train import *
import numpy as np
import scipy.io as io
from PIL import Image
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def test(modelConfig):
    # cude or cpu
    if modelConfig['cpu-cuda'] & torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # load data
    hs, gt, min_value, max_value = load_test_data(modelConfig)
    BL = RDP_process(modelConfig)
    hs = hs.to(device)
    BL = BL.to(device)

    # load models
    d_model = modelConfig['sub_band'] // modelConfig['n_clip']
    path_model_pth = './pth/' + modelConfig['dataset'] + '-sub_band-' + str(modelConfig['sub_band']) + '/'
    Transformer_Encoder = TransformerEncoder(d_model, modelConfig['n_heads'], modelConfig['d_ff'],
                                              modelConfig['n_layers']).to(device)
    Transformer_Encoder.load_state_dict(torch.load(path_model_pth + 'Transformer_Encoder.pth', map_location=device))
    CCN = CombineCov_Net(modelConfig['n_clip'], modelConfig['sub_band']).to(device)
    CCN.load_state_dict(torch.load(path_model_pth + 'CCN.pth', map_location=device))

    # processing
    RDP_sample_CDF = RDP_Sample(BL, modelConfig['sub_band'], device)
    RDP_sample_CDF = RDP_sample_CDF.view(RDP_sample_CDF.shape[0], modelConfig['n_clip'], d_model)
    hs = hs.view(hs.shape[0], modelConfig['n_clip'], d_model)

    start_time = time.time()
    hs_encode = Transformer_Encoder(hs)
    RDP_Sample_encode = Transformer_Encoder(RDP_sample_CDF)
    Pred = CCN(RDP_Sample_encode, hs_encode)
    end_time = time.time()

    # 运行时间
    elapsed_time = end_time - start_time
    print(f"代码运行时间为：{elapsed_time} 秒/n")
    Pred_img, PF, PD, threshold, AUC_list = save_test_results(Pred.detach().view(Pred.shape[0]), gt, modelConfig, device)
    
    return


def load_test_data(modelConfig):
    sub_band = str(modelConfig['sub_band'])
    data_path = './data/sub-band-' + sub_band + '/' + modelConfig['dataset'] + '-' + sub_band + '.mat'
    mat = io.loadmat(data_path)
    hs = mat['data'].astype(np.float32)
    gt = mat['map'].astype(np.float32)
    hs = torch.tensor(hs)
    gt = torch.tensor(gt)

    # normalize
    min_value = hs.min()
    max_value = hs.max()
    hs = hs / 1024
    # to vector
    hs_v = hs.contiguous().view(-1, hs.shape[-1])

    return hs_v, gt, min_value, max_value

def save_test_results(Pred, gt, modelConfig, device):
    if device == 'cuda':
        Pred = Pred.cpu()
    Pred_img = np.array(Pred.reshape(gt.size()))
    gt = gt.reshape(Pred.size())
    Pred = np.array(Pred)
    gt = np.array(gt)
    Pred = (Pred - Pred.min()) / (Pred.max() - Pred.min())
    PF, PD, threshold = roc_curve(gt, Pred)

    save_path = './results/test/' + modelConfig['dataset'] + '-sub_band-' + str(modelConfig['sub_band']) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # ------------------------------------
    # ------------------------------------save img
    Pred_img = ((Pred_img - Pred_img.min()) / (Pred_img.max() - Pred_img.min()) * 255).astype(np.uint8)
    Pred_img = Image.fromarray(Pred_img)
    Pred_img.save(save_path + 'Pred_img.png')
    # ------------------------------------
    # ------------------------------------AUC
    AUC_PDPF = round(auc(PF, PD), 5)
    AUC_PFT = round(auc(threshold, PF), 5)
    AUC_PDT = round(auc(threshold, PD), 5)
    AUC_OA = round(AUC_PDPF + AUC_PDT - AUC_PFT, 5)
    AUC_SNPR = round(AUC_PDT/AUC_PFT, 5)

    AUC_list = {
        'AUC_PDPF': AUC_PDPF,
        'AUC_PFT':  AUC_PFT,
        'AUC_PDT':  AUC_PDT,
        'AUC_OA':   AUC_OA,
        'AUC_SNPR': AUC_SNPR,
    }
    pprint(AUC_list)
    plt.figure()
    plt.imshow(Pred_img)

    plt.figure()
    plt.semilogx(PF, PD)

    return Pred_img, PF, PD, threshold, AUC_list



if __name__ == '__main__':
    modelConfig = {
        'state':            'test',    # train or test
        'dataset':          'Sandiego', # name of the dataset
        'epoch':            1000,       #
        'alpha':            0.2,        # margin
        'n_heads':          2,          # number of multi-head attention
        'd_ff':             512,        # Dimension of feedforward
        'n_layers':         2,          # number of transformer layer
        'n_clip':           5,          # number of clips
        'sub_band':         100,         # number of band in CS
        'cpu-cuda':         1,          # 0-cpu 1-cuda
    }
    test(modelConfig)