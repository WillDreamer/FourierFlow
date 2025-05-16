import sys
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import argparse
from functools import reduce
from functools import partial
from timeit import default_timer
from pdebench_datasets import PDEBench, PDEBench_npy
# from utils import FNODatasetSingle, FNODatasetMult
from metrics import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training(if_training = True,
                 continue_training = False,
                 gamma = 1,
                 num_workers = 4,
                 modes = 12,
                 width = 20,
                 initial_step = 10,
                 t_train = 21,
                 num_channels = 4,
                 batch_size = 64,
                 epochs = 500,
                 learning_rate = 1.e-3,
                 scheduler_step = 100,
                 scheduler_gamma = 0.5,
                 model_update = 2,
                 flnm = ['2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'],
                 single_file = True,
                 reduced_resolution = 1,
                 reduced_resolution_t = 1,
                 reduced_batch = 1,
                 plot = False,
                 channel_plot = 0,
                 x_min = -1,
                 x_max = 1,
                 y_min = -1,
                 y_max = 1,
                 t_min = 0,
                 t_max = 5,
                 base_path='your/path',
                 training_type='autoregressive'
                 ):
    
    ################################################################
    # load data
    ################################################################
    

    # filename
    model_name = flnm[0][:-5] + '_sit_predict'
    # print("FNODatasetSingle")

    # Initialize the dataset and dataloader
    num_initial_steps = 5
    num_future_steps = 5
    train_data = PDEBench_npy(
        data_path=base_path,
        filename=flnm,
        num_frames=num_initial_steps+num_future_steps,
        normalize=True,
        use_spatial_sample=True,
        use_coordinates=False,
        is_train=True,
    )
    val_data = PDEBench_npy(
        data_path=base_path,
        filename=flnm,
        num_frames=num_initial_steps+num_future_steps,
        normalize=True,
        use_spatial_sample=True,
        use_coordinates=False,
        is_train=False,
    )
       

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    
    _data = next(iter(val_loader))['video']
    print(_data.shape,'********')
    _data = next(iter(train_loader))['video']
    print(_data.shape)
                   
    return loss_dict


            
if __name__ == "__main__":

    sum_dict = {
    'L2_best': 0,
    'RMSE_best': 0,
    'nRMSE_best': 0,
    'CSV_best': 0,
    'Max_best': 0,
    'BD_best': 0,
    'F0_best': 0,
    'F1_best': 0,
    'F2_best': 0}
    
    seeds = [42]
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        loss_dict = run_training()
    
        for key in sum_dict:
            sum_dict[key] += loss_dict[key]

    avg_dict = {key: value / len(seeds) for key, value in sum_dict.items()}

    print("=="*5, "Final Average values:", avg_dict)


        

