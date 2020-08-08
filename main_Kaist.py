import os
import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = '/media/mines/46230797-4d43-4860-9b76-ce35e699ea47/KAIST' #TO EDIT
address = os.path.join(base_dir, 'results/Kaist/2020_08_06_14_17_25') #TO EDIT
#address = "last"
################################################################################
# Network parameters
################################################################################
net_class = sn.IMUNet
net_params = {
    'in_dim': 9,
    'out_dim': 1,
    'c0': 16,
    'dropout': 0.1,
    'ks': [7, 7, 7, 7],
    'ds': [4, 4, 4],
    'momentum': 0.1,
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
        'urban06',
        'urban09',
        'urban10',
        'urban11',
        'urban12',
        'urban13',
        'urban15',
        'urban17',
        'urban18',
        'urban19',
        'urban26',
        'urban27',
        'urban28',
        'urban30',
        'urban31',
        'urban33',
        'urban34',
        'urban35',
        'urban36',
        'urban38',
        'urban39',
        ],
    'val_seqs': [
        'urban07',
        'urban14',
        #'urban16',
        ],
    'test_seqs': [
        #'urban07',
        #'urban14',
        'urban16',
        ],
    'dt': 0.01,
}
################################################################################
# Training parameters
################################################################################
train_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01,
        'weight_decay': 1e-2,
    },
    'loss_class': sl.VLoss,
    'loss': {
        'pos_weight': 7,
        'n0': 448, # max dilation
        'w':  1e2,
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 600,
        'T_mult': 2,
        'eta_min': 1e-4,
    },
    'dataloader': {
        'batch_size': 1,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
    },
    # frequency of validation step
    'freq_val': 10,
    # total number of epochs
    'n_epochs': 3600,
    # where record results ?
    'res_dir': os.path.join(base_dir, "results/Kaist"),
    # where record Tensorboard log ?
    'tb_dir': os.path.join(base_dir, "results/runs/Kaist"),
}
################################################################################
# Train on training data set
################################################################################
learning_process = lr.ZUPTProcessing(train_params['res_dir'],
    train_params['tb_dir'], net_class, net_params, None, dataset_params['dt'])
learning_process.train(dataset_class, dataset_params, train_params)
################################################################################
# Test on full data set
################################################################################
learning_process = lr.ZUPTProcessing(train_params['res_dir'],
    train_params['tb_dir'], net_class, net_params, address,
    dataset_params['dt'])
learning_process.test(dataset_class, dataset_params, ['test'])
