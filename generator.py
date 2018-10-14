#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:03:23 2018

@author: sebastian
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import warp,AffineTransform

class Generator(Dataset):
    def __init__(self, X, Y, batch_size, n_frames = 5,size_x = None ,size_y = None ,randomize = True, rotate = True, mirror = True):
        self.X = X
        self.Y = Y 

        n_categories = len(Y[0].keys())
        
        if not size_x:
            [self.size_x,self.size_y] = X.shape[2:-1]
        else:
            [self.size_x,self.size_y] = [size_x,size_y]
        input_size = (3,n_frames)+ tuple([self.size_x,self.size_y] )   
        self.input_x = np.zeros((batch_size,)+input_size)
        self.input_y = np.zeros([batch_size,n_categories]+[self.size_x//2,self.size_y//2])#+X.shape[2:-1])        
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.idcs = np.arange(X.shape[0]-input_size[-1]-1)
        if randomize:
            self.idcs = np.random.permutation(self.idcs)
        self.idx = 0  
    def __len__(self):
        return(self.batch_size)
    def __getitem__(self, batch = None):
        self.input_y[:,:,:,:] = 0
        for sample in range(self.batch_size):
            
            n = self.idcs[self.idx]
            
            scale=1+np.random.normal()/10
            rotation=np.random.randint(4)#np.random.uniform()*2
            translation = [([0,1,1,0][rotation])*self.X.shape[2],(([0,0,1,1][rotation]))*self.X.shape[3]]
            translation[0]+=0 #np.random.randint(self.X.shape[2]-self.size_x)
            translation[1]+=0 #np.random.randint(self.X.shape[3]-self.size_y)
            rotation *= np.pi/2 +np.random.normal()/50
            
            tform = AffineTransform(scale=[scale]*2, rotation=rotation,translation = translation)
            
            start = 5-(self.n_frames//2)
            for img in range(self.n_frames):
                self.input_x[sample,:,img,:,:]  = warp(self.X[n,start+img,:,:,:]/255, tform.inverse, output_shape=(self.size_x, self.size_y)).transpose([2,0,1])
                
            
            for j,label in enumerate(self.Y[n].keys()):
                x_s = [(l) for k,l in enumerate(self.Y[n][label][0])]
                y_s = [(l) for k,l in enumerate(self.Y[n][label][1])]
                for k in range(len(x_s)):
                    coords = tform([x_s[k],y_s[k]]).astype(np.int)[0]
                    if all(coords>=0) and all (coords<self.size_x):#all((self.X.shape[2:4][::-1]-coords)>0):     
                        self.input_y[sample,j,coords[1]//2,coords[0]//2]=1
                            
            
            self.idx = (self.idx+1)%(len(self.idcs))
        return(torch.from_numpy(self.input_x).float().cuda(),torch.from_numpy(self.input_y).float().cuda())
        
