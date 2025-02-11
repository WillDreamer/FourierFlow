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
                 base_path='/storage/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/',
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

    # dimensions = len(_data.shape)
    # print('Spatial Dimension', dimensions - 3)
    # if dimensions == 4:
    #     model = FNO1d(num_channels=num_channels,
    #                   width=width,
    #                   modes=modes,
    #                   initial_step=initial_step).to(device)
    # elif dimensions == 5:
    #     model = FNO2d(num_channels=num_channels,
    #                   width=width,
    #                   modes1=modes,
    #                   modes2=modes,
    #                   initial_step=initial_step,
    #                   gamma = gamma).to(device)
    # elif dimensions == 6:
    #     model = FNO3d(num_channels=num_channels,
    #                   width=width,
    #                   modes1=modes,
    #                   modes2=modes,
    #                   modes3=modes,
    #                   initial_step=initial_step).to(device)
        
    # # Set maximum time step of the data to train
    # if t_train > _data.shape[-2]:
    #     t_train = _data.shape[-2]

    # model_path = model_name + ".pt"
    
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total parameters = {total_params}')
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    # loss_fn = nn.MSELoss(reduction="mean")
    # loss_val_min = np.infty
    # RMSE_val_min = np.infty
    # nRMSE_val_min = np.infty
    # CSV_val_min = np.infty
    # Max_val_min = np.infty
    # BD_val_min = np.infty
    # F0_val_min = np.infty
    # F1_val_min = np.infty
    # F2_val_min = np.infty
    
    # start_epoch = 0

    # if not if_training:
    #     checkpoint = torch.load(model_path, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model.to(device)
    #     model.eval()
    #     Lx, Ly, Lz = 1., 1., 1.
    #     errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
    #                    model_name, x_min, x_max, y_min, y_max,
    #                    t_min, t_max, initial_step=initial_step)
    #     pickle.dump(errs, open(model_name+'.pickle', "wb"))
        
    #     return

    # # If desired, restore the network by loading the weights saved in the .pt
    # # file
    # if continue_training:
    #     print('Restoring model (that is the network\'s weights) from file...')
    #     checkpoint = torch.load(model_path, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model.to(device)
    #     model.train()
        
    #     # Load optimizer state dict
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.to(device)
                    
    #     start_epoch = checkpoint['epoch']
    #     loss_val_min = checkpoint['loss']
    
    # for ep in range(start_epoch, epochs):
    #     model.train()
    #     t1 = default_timer()
    #     train_l2_step = 0
    #     train_l2_full = 0
    #     for xx, yy, grid in train_loader:
    #         loss = 0
            
    #         # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
    #         # yy: target tensor [b, x1, ..., xd, t, v]
    #         # grid: meshgrid [b, x1, ..., xd, dims]
    #         xx = xx.to(device)
    #         yy = yy.to(device)
    #         grid = grid.to(device)

    #         # Initialize the prediction tensor
    #         pred = yy[..., :initial_step, :]
    #         # Extract shape of the input tensor for reshaping (i.e. stacking the
    #         # time and channels dimension together)
    #         inp_shape = list(xx.shape)
    #         inp_shape = inp_shape[:-2]
    #         inp_shape.append(-1)
    
    #         if training_type in ['autoregressive']:
    #             # Autoregressive loop
    #             for t in range(initial_step, t_train):
                    
    #                 # Reshape input tensor into [b, x1, ..., xd, t_init*v]
    #                 inp = xx.reshape(inp_shape)
                    
    #                 # Extract target at current time step
    #                 y = yy[..., t:t+1, :]

    #                 # Model run
    #                 im = model(inp, grid)

    #                 # Loss calculation
    #                 _batch = im.size(0)
    #                 loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
        
    #                 # Concatenate the prediction at current time step into the
    #                 # prediction tensor
    #                 pred = torch.cat((pred, im), -2)
        
    #                 # Concatenate the prediction at the current time step to be used
    #                 # as input for the next time step
    #                 xx = torch.cat((xx[..., 1:, :], im), dim=-2)

    #             train_l2_step += loss.item()
    #             _batch = yy.size(0)
    #             _yy = yy[..., :t_train, :]  # if t_train is not -1
    #             l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
    #             train_l2_full += l2_full.item()
        
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #         if training_type in ['single']:
    #             x = xx[..., 0 , :]
    #             y = yy[..., t_train-1:t_train, :]
    #             pred = model(x, grid)
    #             _batch = yy.size(0)
    #             loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
    
    #             train_l2_step += loss.item()
    #             train_l2_full += loss.item()
        
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #     if ep % model_update == 0:
    #         val_l2_step = 0
    #         val_l2_full = 0
    #         err_RMSE = 0
    #         err_nRMSE = 0
    #         err_CSV = 0
    #         err_Max = 0
    #         err_BD = 0
    #         err_F0 = 0
    #         err_F1 = 0
    #         err_F2 = 0
    #         with torch.no_grad():
    #             for xx, yy, grid in val_loader:
    #                 loss = 0
    #                 xx = xx.to(device)
    #                 yy = yy.to(device)
    #                 grid = grid.to(device)
                    
    #                 if training_type in ['autoregressive']:
    #                     pred = yy[..., :initial_step, :]
    #                     inp_shape = list(xx.shape)
    #                     inp_shape = inp_shape[:-2]
    #                     inp_shape.append(-1)
                
    #                     for t in range(initial_step, yy.shape[-2]):
    #                         inp = xx.reshape(inp_shape)
    #                         y = yy[..., t:t+1, :]
    #                         im = model(inp, grid)
    #                         _batch = im.size(0)
    #                         loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

    #                         pred = torch.cat((pred, im), -2)
                
    #                         xx = torch.cat((xx[..., 1:, :], im), dim=-2)
            
    #                     val_l2_step += loss.item()
    #                     _batch = yy.size(0)
    #                     _pred = pred[..., initial_step:t_train, :]
    #                     _yy = yy[..., initial_step:t_train, :]
    #                     val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
    #                     Lx, Ly, Lz = 1., 1., 1.
    #                     _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
    #                     = metric_func(pred, yy, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)
    #                     err_RMSE += _err_RMSE
    #                     err_nRMSE += _err_nRMSE
    #                     err_CSV += _err_CSV
    #                     err_Max += _err_Max
    #                     err_BD += _err_BD
    #                     err_F0 += _err_F[0]
    #                     err_F1 += _err_F[1]
    #                     err_F2 += _err_F[2]

    #                 if training_type in ['single']:
    #                     x = xx[..., 0 , :]
    #                     y = yy[..., t_train-1:t_train, :]
    #                     pred = model(x, grid)
    #                     _batch = yy.size(0)
    #                     loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
            
    #                     val_l2_step += loss.item()
    #                     val_l2_full += loss.item()
                
    #             if  val_l2_full < loss_val_min:
    #                 loss_val_min = val_l2_full
    #                 RMSE_val_min = err_RMSE.item()
    #                 nRMSE_val_min = err_nRMSE.item()
    #                 CSV_val_min = err_CSV.item()
    #                 Max_val_min = err_Max.item()
    #                 BD_val_min = err_BD.item()
    #                 F0_val_min = err_F0.item()
    #                 F1_val_min = err_F1.item()
    #                 F2_val_min = err_F2.item()
    #                 torch.save({
    #                     'epoch': ep,
    #                     'model_state_dict': model.state_dict(),
    #                     'optimizer_state_dict': optimizer.state_dict(),
    #                     'loss': loss_val_min
    #                     }, model_path)
                
            
    #     t2 = default_timer()
    #     scheduler.step()
    #     loss_dict = {
    #         'L2_best': loss_val_min,
    #         'RMSE_best':RMSE_val_min,
    #         'nRMSE_best':nRMSE_val_min,
    #         'CSV_best':CSV_val_min,
    #         'Max_best':Max_val_min,
    #         'BD_best':BD_val_min,
    #         'F0_best':F0_val_min,
    #         'F1_best':F1_val_min,
    #         'F2_best':F2_val_min
    #     }
    #     if (ep+1) % 50 ==0:
    #         print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}, trainL2: {3:.5f}, testL2: {4:.5f}, err_RMSE: {5:.5f}, err_nRMSE: {6:.5f}, err_CSV: {7:.5f}, err_Max: {8:.5f}, err_BD: {9:.5f},err_F0: {10:.5f}, err_F1: {11:.5f}, err_F2: {12:.5f}'\
    #             .format(ep, loss.item(), t2 - t1, train_l2_full, val_l2_full, err_RMSE.item(), err_nRMSE.item(), err_CSV.item(), err_Max.item(), err_BD.item(),err_F0.item(),err_F1.item(),err_F2.item()))
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
    
        # 将当前字典的值累加到总和字典中
        for key in sum_dict:
            sum_dict[key] += loss_dict[key]

    # 计算平均值
    avg_dict = {key: value / len(seeds) for key, value in sum_dict.items()}

    # 打印平均值字典
    print("=="*5, "Final Average values:", avg_dict)


        

