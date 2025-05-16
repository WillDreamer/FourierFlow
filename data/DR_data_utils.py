
import copy
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt
import random

class GaussianNormalizer(object):
    def __init__(self, x, eps=1e-8):
        super(GaussianNormalizer, self).__init__()
        
        # x: numpy array of shape [b, h, w, t, c]
        # Compute mean and std along the first four axes (excluding the channel axis)
        # for each channel separately
        self.mean = np.mean(x, axis=(0, 1, 2, 3), keepdims=True)  # shape will be [1, 1, 1, 1, c]
        self.std = np.std(x, axis=(0, 1, 2, 3), keepdims=True)    # shape will be [1, 1, 1, 1, c]
        self.eps = eps

    def encode(self, x):
        # Normalize each channel separately
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        # De-normalize each channel separately
        x = (x * (self.std + self.eps)) + self.mean
        return x


class FNODatasetSingle(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1,
                 logger = ''
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        # self.data = torch.randn((100,128,128,21,4))
        # self.grid = torch.ones((128,128,2))
        # Define path to files
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        if filename[-2:] != 'h5':
            logger.info(f".HDF5 file extension is assumed hereafter")
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                logger.info(keys)
                keys.sort()
                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    logger.info('Data Dimension: {}D'.format(len(idx_cfd)-2))
                    if len(idx_cfd)==3:  # 1D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                            dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,2] = _data   # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(idx_cfd)==4:  # 2D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,3] = _data   # batch, x, t, ch
                        # self.normalizer = GaussianNormalizer(self.data[:int(idx_cfd[0]//reduced_batch)])
                        # self.data = self.normalizer.encode(self.data)
                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
                
                    if len(idx_cfd)==5:  # 3D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                        self.grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution,\
                                                                    ::reduced_resolution,\
                                                                    ::reduced_resolution]
                                                                    
                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data = _data[:, :, :, None]  # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

        elif filename[-2:] == 'h5':  # SWE-2D (RDB)
            print(f".H5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                data_arrays = [np.array(f[key]['data'], dtype=np.float32) for key in keys]
                _data = torch.from_numpy(np.stack(data_arrays, axis=0))   # [batch, nt, nx, ny, nc]
                _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ...]
                _data = torch.permute(_data, (0, 2, 3, 1, 4))   # [batch, nx, ny, nt, nc]
                gridx, gridy = np.array(f['0023']['grid']['x'], dtype=np.float32), np.array(f['0023']['grid']['y'], dtype=np.float32)
                mgridX, mgridY = np.meshgrid(gridx, gridy, indexing='ij')
                _grid = torch.stack((torch.from_numpy(mgridX), torch.from_numpy(mgridY)), axis=-1)
                grid = _grid[::reduced_resolution, ::reduced_resolution, ...]
                _tsteps_t = torch.from_numpy(np.array(f['0023']['grid']['t'], dtype=np.float32))
                tsteps_t = _tsteps_t[::reduced_resolution_t]
                self.data = _data
                self.grid = _grid
                self.tsteps_t = tsteps_t

        self.initial_step = initial_step

        self.data = self.data if torch.is_tensor(self.data) else torch.tensor(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # return self.data[idx,...,:self.initial_step,:], self.data[idx], self.grid
        
        rand_idx = random.randint(0, int(self.data.shape[-2])-2)
        return self.data[idx,...,rand_idx+1,:], self.grid, self.data[idx,...,rand_idx,:]

    @staticmethod
    def get_train_test_datasets(filename,
                              initial_step=10,
                              saved_folder='../data/',
                              reduced_resolution=1,
                              reduced_resolution_t=1,
                              reduced_batch=1,
                              test_ratio=0.1,
                              num_samples_max=-1,
                              logger=''):
        dataset = FNODatasetSingle(filename,
                                 initial_step=initial_step,
                                 saved_folder=saved_folder,
                                 reduced_resolution=reduced_resolution,
                                 reduced_resolution_t=reduced_resolution_t,
                                 reduced_batch=reduced_batch,
                                 logger=logger)
        
        total_samples = len(dataset.data)
        if num_samples_max > 0:
            total_samples = min(num_samples_max, total_samples)
            
        indices = torch.randperm(total_samples)
        test_size = int(total_samples * test_ratio)
        
        train_dataset = copy.deepcopy(dataset)
        test_dataset = copy.deepcopy(dataset)

        train_dataset.data = dataset.data[indices[test_size:total_samples]]
        test_dataset.data = dataset.data[indices[:test_size]]
        
        
        return train_dataset, test_dataset


class FNODatasetMultistep(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1,
                 if_noise=False
                 ):

        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        if filename[-2:] != 'h5':
            print(f".HDF5 file extension is assumed hereafter")
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                print(keys)
                keys.sort()
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                print('Data Dimension: {}D'.format(len(idx_cfd)-2))
                self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                        idx_cfd[2]//reduced_resolution,
                                        idx_cfd[3]//reduced_resolution,
                                        mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                        4],
                                        dtype=np.float32)
                # density
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data, (0, 2, 3, 1))
                self.data[...,0] = _data   # batch, x, t, ch
                # pressure
                _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data, (0, 2, 3, 1))
                self.data[...,1] = _data   # batch, x, t, ch
                # Vx
                _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data, (0, 2, 3, 1))
                self.data[...,2] = _data   # batch, x, t, ch
                # Vy
                _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data, (0, 2, 3, 1))
                self.data[...,3] = _data   # batch, x, t, ch
                x = np.array(f["x-coordinate"], dtype=np.float32)
                y = np.array(f["y-coordinate"], dtype=np.float32)
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
        elif filename[-2:] == 'h5':
            print(f".H5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                data_arrays = [np.array(f[key]['data'], dtype=np.float32) for key in keys]
                _data = torch.from_numpy(np.stack(data_arrays, axis=0))   # [batch, nt, nx, ny, nc]
                _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ...]
                _data = torch.permute(_data, (0, 2, 3, 1, 4))   # [batch, nx, ny, nt, nc]
                gridx, gridy = np.array(f['0023']['grid']['x'], dtype=np.float32), np.array(f['0023']['grid']['y'], dtype=np.float32)
                mgridX, mgridY = np.meshgrid(gridx, gridy, indexing='ij')
                _grid = torch.stack((torch.from_numpy(mgridX), torch.from_numpy(mgridY)), axis=-1)
                grid = _grid[::reduced_resolution, ::reduced_resolution, ...]
                _tsteps_t = torch.from_numpy(np.array(f['0023']['grid']['t'], dtype=np.float32))
                tsteps_t = _tsteps_t[::reduced_resolution_t]
                self.data = _data
                self.grid = grid
                self.tsteps_t = tsteps_t

        self.initial_step = initial_step
        if if_noise:
            self.data += 0.1 * torch.randn_like(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # return self.data[idx,...,:self.initial_step,:], self.data[idx], self.grid
        k = 4
        rand_idx = random.randint(0, int(self.data.shape[-2])-9)
        return self.data[idx,...,rand_idx+k:rand_idx+k*2,:], self.grid, self.data[idx,...,rand_idx:rand_idx+k,:]

    @staticmethod
    def get_train_test_datasets(filename,
                              initial_step=10,
                              saved_folder='../data/',
                              reduced_resolution=1,
                              reduced_resolution_t=1,
                              reduced_batch=1,
                              test_ratio=0.1,
                              num_samples_max=-1,
                              if_eval_plot = False
                              ):

        dataset = FNODatasetMultistep(filename,
                                 initial_step=initial_step,
                                 saved_folder=saved_folder,
                                 reduced_resolution=reduced_resolution,
                                 reduced_resolution_t=reduced_resolution_t,
                                 reduced_batch=reduced_batch,
                                 )
        
        total_samples = len(dataset.data)
        if num_samples_max > 0:
            total_samples = min(num_samples_max, total_samples)
            
        indices = torch.randperm(total_samples)
        test_size = int(total_samples * test_ratio)
        
        train_dataset = copy.deepcopy(dataset)
        test_dataset = copy.deepcopy(dataset)

        train_dataset.data = dataset.data[indices[test_size:total_samples]]
        normalizer = GaussianNormalizer(train_dataset.data.numpy())
        train_dataset.data = torch.tensor(normalizer.encode(train_dataset.data.numpy()))
        
        test_dataset.data = dataset.data[indices[:test_size]]
        test_dataset.data = torch.tensor(normalizer.encode(test_dataset.data.numpy()))
        
        if if_eval_plot:
            return train_dataset, test_dataset, normalizer
        else:
            return train_dataset, test_dataset


if __name__ == '__main__':
    # flnm = 'ns_incom_inhom_2d_512-28.h5'
    flnm = '2D_diff-react_NA_NA.h5'
    base_path='path_to/data/2D/diffusion-reaction/'

    train_dataset, test_dataset = FNODatasetMultistep.get_train_test_datasets(
                                    flnm,
                                    reduced_resolution=1,
                                    reduced_resolution_t=1,
                                    reduced_batch=1,
                                    initial_step=0,
                                    saved_folder=base_path,
                                )
    # s
    
# 

    
