#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 01:21:21 2018

@author: sebastian
"""

import torch
from torch import nn
from torchvision import models as tmodels
import models.resnet as local_models
   

def create_model(n_classes=5): 
    model = local_models.resnet18(sample_size = None,
                             sample_duration = 5,#file['X'].shape[1], 
                             num_classes = n_classes,
                             include_top = False,
                             stride = 1,                        
                             stride_pool = 2
                             )
    
    pretrained = torch.load('models/resnet-18-kinetics-ucf101_split1.pth')
    state_dict = {}
    model_dict = model.state_dict()
    
    for k in pretrained['state_dict'].keys():
         state_dict[k[7:]] = pretrained['state_dict'][k]
    
    del state_dict['fc.weight'], state_dict['fc.bias']
        
    model_dict.update(state_dict)
    model.load_state_dict(model_dict) 

    
    resnet18 = tmodels.resnet18(pretrained=True, )
    
    class Inter_Layers(nn.Module):
        def __init__(self):
            super(Inter_Layers, self).__init__()
            self.conv1 = nn.Conv2d(64, 64, 3, padding = 1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
            
        def forward(self, x):
            x = x.max(dim = 2, keepdim=False)[0]
            return x
        
        
        
    class Resnet_Hybrid(nn.Module):
        def __init__(self, output_categories):
            super(Resnet_Hybrid, self).__init__()
            self.res2d_pruned = nn.Sequential(*list(resnet18.children())[2:-5])
            self.res3d_pruned = nn.Sequential(*(list(model.children())[:5]))
            self.inter_layers = Inter_Layers()
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') 
            self.conv1 = nn.Conv2d(64, 32, 3, padding = 1)
            self.conv2 = nn.Conv2d(32, output_categories, 1)
            
        def forward(self, x):
            
            x = self.res3d_pruned(x)
            x1 = self.inter_layers(x)
            x = self.res2d_pruned(x1)
            x = self.upsample(x)
            x = x1.add(x[:,:,:x1.shape[2],:x1.shape[2]])
            x = nn.functional.relu(self.conv1(x))
            return nn.functional.sigmoid(self.conv2(x))
        
        
    return Resnet_Hybrid(5)
