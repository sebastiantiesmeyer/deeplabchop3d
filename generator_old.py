#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:03:23 2018

@author: sebastian
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class Generator(Dataset):
    def __init__(self, file, batch_size, idcs):
        
        self.X = file['X']
        self.Y = file['Y'] 
        input_size = self.X.shape[1:]
        output_size = self.Y.shape[1:]
        self.input_x = np.zeros((batch_size,)+input_size)
        self.input_y = np.zeros((batch_size,1)+output_size)        
        self.batch_size = batch_size
        self.input_size = input_size
        self.idcs = idcs
        self.idx = 0  
    def __len__(self):
        return(self.batch_size)
    def __getitem__(self, batch = None):
        for batch in range(self.batch_size):
            self.input_x[batch,:,:,:,:]  = self.X[self.idcs[self.idx],:,:,:]
            self.input_y[batch,0,:,:,:]      = self.Y[self.idcs[self.idx],:,:,:] 
            self.idx = (self.idx+1)%(len(self.idcs))
        return(torch.from_numpy(self.input_x.transpose([0,4,1,2,3]).astype(np.float32)).contiguous(),
               torch.from_numpy(self.input_y.transpose((0,4,1,2,3)).astype(np.float32)).contiguous())
        
   
class Generator1(Dataset):
    def __init__(self, file, batch_size, idcs, n_frames = 10):
        self.n_frames = n_frames
        self.X = file['X']
        self.Y = file['Y'] 
        input_size = self.X.shape[1:]
        output_size = self.Y.shape[1:]
        self.input_x = np.zeros((batch_size,n_frames)+input_size[1:])
        self.input_y = np.zeros((batch_size,1)+output_size)        
        self.batch_size = batch_size
        self.input_size = input_size
        self.idcs = idcs
        self.idx = 0  
    def __len__(self):
        return(self.batch_size)
    def __getitem__(self, batch = None):
        for batch in range(self.batch_size):
            self.input_x[batch,:,:,:,:]  = self.X[self.idcs[self.idx],5-(self.n_frames//2):11-(5-(self.n_frames//2)),:,:,:]
            self.input_y[batch,0,:,:,:]      = self.Y[self.idcs[self.idx],:,:,:] 
            self.idx = (self.idx+1)%(len(self.idcs))
        return(torch.from_numpy(self.input_x.transpose([0,4,1,2,3]).astype(np.float32)/255).contiguous(),
               torch.from_numpy(self.input_y.transpose((0,1,4,2,3)).astype(np.float32)/255).contiguous())
        
