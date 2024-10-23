import torch
import os
from model import *
from utils import *
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

def train(modelConfig):
    torch.autograd.set_detect_anomaly(True)
    # cude or cpu
    if modelConfig['cpu-cuda'] & torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # parameters
    alpha = torch.tensor(modelConfig['alpha']).to(device)
    d_model = modelConfig['sub_band'] // modelConfig['n_clip']
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(log_dir='/root/tf-logs/' + time_stamp)  # save logs to the path

    # get data
    spectral, [min_value, max_value] = data_input_process(modelConfig)
    target_spectral = spectral[1]
    bkg_spectral = spectral[2]

    # get BL
    BL = RDP_process(modelConfig)
    BL = BL.to(device)

    # start training
    print("=====> Constitute Transformer")
    Transformer_Encoder = TransformerEncoder(d_model, modelConfig['n_heads'], modelConfig['d_ff'], modelConfig['n_layers']).to(device)
    CCN = CombineCov_Net(modelConfig['n_clip'], modelConfig['sub_band']).to(device)

    print("=====> Setup optimizer")
    Optimizer_Trans_Encoder = optim.SGD(Transformer_Encoder.parameters(), lr=0.0001, weight_decay=1e-2)
    Optimizer_CCN = optim.SGD(CCN.parameters(), lr=0.0008, weight_decay=1e-2)

    print("=====> Start Training")
    target_spectral = target_spectral.view(target_spectral.shape[0], modelConfig['n_clip'], d_model)
    bkg_spectral = bkg_spectral.view(bkg_spectral.shape[0], modelConfig['n_clip'], d_model)

    BCE_loss_fun = nn.BCELoss()
    k = 10
    P_label = torch.ones(1, 1, 1).to(device)
    N_label = torch.zeros(k, 1, 1).to(device)
    Triplet_Loss_ = []
    BCE_loss_ = []
    for epoch in range(modelConfig['epoch']):
        Triplet_loss = torch.tensor(0).to(device)
        BCE_loss = torch.tensor(0).to(device)
        for i in range(target_spectral.shape[0]):
        # for positive in data_loader:
            Transformer_Encoder.zero_grad()
            CCN.zero_grad()
            # ------------------------------------
            # ------------------------------------
            positive = target_spectral[i, :, :]
            positive = positive.unsqueeze(0)
            anchor = RDP_Sample(BL, modelConfig['sub_band'], device)
            anchor = anchor.view(anchor.shape[0], modelConfig['n_clip'], d_model)
            # ------------------------------------
            # ------------------------------------
            positive_encode = Transformer_Encoder(positive)
            anchor_encode = Transformer_Encoder(anchor)
            bkg_encode = Transformer_Encoder(bkg_spectral)
            # ------------------------------------
            # ------------------------------------
            Triplet_loss_i, hard_neg = Triplet_Loss_fun(positive_encode, bkg_encode, anchor_encode, alpha)
            if hard_neg.shape[0] == 0:
                continue
            (Triplet_loss_i).backward()
            # ------------------------------------
            # ------------------------------------
            positive_pred = CCN(anchor_encode.detach(), positive_encode.detach())
            hard_neg_pred = CCN(anchor_encode.detach(), hard_neg.detach())
            # ------------------------------------
            # ------------------------------------
            BCE_loss_i = (BCE_loss_fun(positive_pred, P_label) * 1 + BCE_loss_fun(hard_neg_pred, torch.zeros_like(hard_neg_pred)) * 1)/2
            (BCE_loss_i).backward()

            Optimizer_Trans_Encoder.step()
            Optimizer_CCN.step()

            Triplet_loss = Triplet_loss_i + Triplet_loss
            BCE_loss = BCE_loss_i + BCE_loss
            # ------------------------------------
            # ------------------------------------
        Triplet_loss = Triplet_loss / target_spectral.shape[0]
        BCE_loss = BCE_loss / target_spectral.shape[0]
        # ------------------------------------
        # ------------------------------------ save training data
        writer.add_scalar('Triplet_Loss', Triplet_loss.detach(), epoch)
        writer.add_scalar('BCE_loss', BCE_loss.detach(), epoch)
        Triplet_Loss_.append(Triplet_loss.detach())
        BCE_loss_.append(BCE_loss.detach())
        if epoch % 10 == 0:
            print('[%d/%d] Triplet_Loss: %.4f BCE_loss: %.4f '
                  % (epoch, modelConfig['epoch'], Triplet_loss, BCE_loss))
    CCN = CCN_Fin_Tuning(target_spectral, bkg_spectral, Transformer_Encoder, CCN, modelConfig, BL, device, writer)
    # save training results
    if device == 'cuda':
        Triplet_Loss_ = [loss.cpu() for loss in Triplet_Loss_]
        BCE_loss_ = [loss.cpu() for loss in BCE_loss_]

    path_model_pth = './pth/' + modelConfig['dataset'] + '-sub_band-' + str(modelConfig['sub_band']) + '/'
    path_train_result = './results/train/' + modelConfig['dataset'] + '-sub_band-' + str(modelConfig['sub_band']) + '/'

    if not os.path.exists(path_model_pth):
        os.makedirs(path_model_pth)
    if not os.path.exists(path_train_result):
        os.makedirs(path_train_result)

    plt.figure()
    plt.plot(Triplet_Loss_, label='Triplet_Loss')
    plt.plot(BCE_loss_, label='BCE_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_train_result + 'loss_KLD.png')

    torch.save(Transformer_Encoder.state_dict(), path_model_pth + 'Transformer_Encoder.pth')
    torch.save(CCN.state_dict(), path_model_pth + 'CCN.pth')

    return 0
def CCN_Fin_Tuning(target_spectral, bkg_spectral, Transformer_Encoder, CCN, modelConfig, BL, device, writer):
    Optimizer_CCN = optim.Adam(CCN.parameters(), lr=0.0005)
    d_model = modelConfig['sub_band'] // modelConfig['n_clip']
    P_label = torch.ones(1, 1, 1).to(device)
    BCE_loss_ = []
    BCE_loss_fun = nn.BCELoss()
    # ------------------------------------
    # ------------------------------------
    for epoch in range(modelConfig['epoch']):
        BCE_loss = torch.tensor(0).to(device)
        for i in range(target_spectral.shape[0]):
            CCN.zero_grad()
            positive = target_spectral[i, :, :]
            positive = positive.unsqueeze(0)

            anchor = RDP_Sample(BL, modelConfig['sub_band'], device)
            anchor = anchor.view(anchor.shape[0], modelConfig['n_clip'], d_model)
            # ------------------------------------
            # ------------------------------------
            positive_encode = Transformer_Encoder(positive)
            anchor_encode = Transformer_Encoder(anchor)
            bkg_encode = Transformer_Encoder(bkg_spectral)
            # ------------------------------------
            # ------------------------------------
            simehard_neg = SemiHard_Neg_Minning(positive_encode, bkg_encode, anchor_encode)
            if simehard_neg.shape[0] == 0:
                continue
            # ------------------------------------
            # ------------------------------------
            positive_pred = CCN(anchor_encode.detach(), positive_encode.detach())
            semihard_neg_pred = CCN(anchor_encode.detach(), simehard_neg.detach())
            # ------------------------------------
            # ------------------------------------
            BCE_loss_i = (BCE_loss_fun(positive_pred, P_label) +
                          BCE_loss_fun(semihard_neg_pred, torch.zeros_like(semihard_neg_pred))) / 2
            (BCE_loss_i).backward()
            Optimizer_CCN.step()
            BCE_loss = BCE_loss_i.detach() + BCE_loss
        BCE_loss = BCE_loss / target_spectral.shape[0]
        writer.add_scalar('BCE_loss', BCE_loss.detach(), epoch + modelConfig['epoch'])
        if epoch % 10 == 0:
            print('Fin-Tunning [%d/%d] BCE_loss: %.4f '
                  % (epoch, modelConfig['epoch'], BCE_loss))
    return CCN

def SemiHard_Neg_Minning(positive_encode, bkg_encode, anchor_encode):
    anchor_like_bkg = anchor_encode.expand_as(bkg_encode)

    Distance_anchor2bkg = torch.norm(anchor_like_bkg - bkg_encode, dim=2)
    Distance_positive2anchor = torch.norm(positive_encode - anchor_encode, dim=2)

    # lower_bound = Distance_positive2anchor - alpha
    upper_bound = Distance_positive2anchor + 0.1
    simehard_neg = bkg_encode[(Distance_anchor2bkg <= upper_bound)]

    return simehard_neg.unsqueeze(1)
def RDP_Sample(BL, band_num, device):

    sample = torch.randn(1, band_num).to(device)
    sample = sample*torch.sqrt(BL)
    sample = sample / 1024
    # RDP_sample_CDF = CDF(sample)
    RDP_sample_CDF = (sample)

    return RDP_sample_CDF

def Triplet_Loss_fun(positive_encode, bkg_encode, anchor_encode, alpha):

    anchor_like_bkg = anchor_encode.expand_as(bkg_encode)

    Distance_anchor2bkg = torch.norm(anchor_like_bkg - bkg_encode, dim=2)
    Distance_positive2anchor = torch.norm(positive_encode - anchor_encode, dim=2)

    # lower_bound = Distance_positive2anchor - alpha
    upper_bound = Distance_positive2anchor
    selected_negative = Distance_anchor2bkg[(Distance_anchor2bkg <= upper_bound)]
    hard_neg = bkg_encode[(Distance_anchor2bkg <= upper_bound)]

    loss = Distance_positive2anchor - selected_negative + alpha
    loss = torch.sum(loss[loss > 0]) / selected_negative.shape[0]

    return loss, hard_neg.unsqueeze(1)

if __name__ == '__main__':
    modelConfig = {
        'state':            'train',    # train or test
        'dataset':          'Sandiego', # name of the dataset
        'epoch':            400,       #
        'alpha':            1,        # margin
        'n_heads':          2,          # number of multi-head attention
        'd_ff':             512,        # Dimension of feedforward
        'n_layers':         2,          # number of transformer layer
        'n_clip':           5,          # number of clips
        'sub_band':         100,         # number of band in CS
        'cpu-cuda':         1,          # 0-cpu 1-cuda
    }
    train(modelConfig)