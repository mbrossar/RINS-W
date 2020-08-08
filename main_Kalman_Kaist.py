import os
import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np
import shutil

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = '/media/mines/46230797-4d43-4860-9b76-ce35e699ea47/KAIST' #TO EDIT
address = os.path.join(base_dir, 'results/Kaist/2020_08_06_14_17_25') #TO EDIT

################################################################################
# Kalman filter parameters
################################################################################
iekf_params = {
    'th_max_zupt': 0.98,
    'th_min_zupt': 0.98,
    'max_omega_norm': 0.1,
    'max_omega': 0.1,
    'max_acc_norm': 0.6,
    'max_acc': 0.6,
    'zupt_omega_std': 0.04,
    'zupt_acc_std': 0.4,
    'N_init': 1000,
    'N_normalize': 10000,
}
net_class = sn.BBBNet
bbb_net_params = {
    'zupt_forward_std': 1,
    'lat_std': 2,
    'up_std': 3,
}
################################################################################
# Dataset parameters
################################################################################
dataset_class = ds.KaistDataset
dataset_params = {
    # where are raw data ?
    'data_dir': data_dir,
    # where record preloaded data ?
    'predata_dir': os.path.join(base_dir, 'data/Kaist'),
    # set train, val and test sequence
    'train_seqs': [
        ],
    'val_seqs': [
        ],
    'test_seqs': [
        'urban07',
        'urban14',
        'urban16',
        ],
    'dt': 0.01,
}
################################################################################
# Training parameters
################################################################################
train_params = {
    # where record results ?
    'res_dir': os.path.join(base_dir, "results/Kaist"),
    # where record Tensorboard log ?
    'tb_dir': os.path.join(base_dir, "results/runs/Kaist"),
    'loss_class': sl.VLoss,
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
    },
    'loss_class': sl.VLoss,
    'loss': {
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
    },
    'dataloader': {
    },
    # frequency of validation step
    'freq_val': 0,
    # total number of epochs
    'n_epochs': 0,
}
################################################################################
# Test on full data set
################################################################################
learning_process = lr.KalmanProcessing(train_params['res_dir'],
    train_params['tb_dir'], net_class, bbb_net_params, address,
    dataset_params['dt'], iekf_params, train_params)
learning_process.test(dataset_class, dataset_params, ['test'])
learning_process.display_test(dataset_class, dataset_params, 'test')