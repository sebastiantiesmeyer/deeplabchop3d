import torch
from torch.autograd import Variable
#import time
import os
#import sys
import h5py
from utils import Logger
import pickle
import generator

from train import train_epoch
from validation import val_epoch
from utils import AverageMeter, calculate_accuracy
from generator import Generator
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from modules import create_model

data_path =('/home/sebastian/code/3D-ResNets-PyTorch/deeplabcut3d/projects/test1-sebastian-2018-09-30/data/')
train_test_split=0.9
n_epochs=100
batch_size = 1
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0001
nesterov = True

file = h5py.File(os.path.join(data_path,'ExtractedFrames.h5'),'r')
n_frames = file['X'].shape[0]
shuffle = np.random.permutation(np.arange(n_frames))
train_test_split = int(shuffle.shape[0]*train_test_split) 

#train_loader = Generator(file,batch_size=batch_size, idcs=shuffle[0:train_test_split])#,n_frames = 5)
#val_loader = Generator(file,batch_size=batch_size, idcs=shuffle[train_test_split:])#,n_frames = 5)

train_logger = Logger(
os.path.join(data_path, 'train.log'),
['epoch', 'loss', 'acc', 'lr'])

train_batch_logger = Logger(
os.path.join(data_path, 'train_batch.log'),
['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

val_logger = Logger(
os.path.join(data_path, 'test.log'),
['epoch', 'loss', 'acc', 'lr'])


#res2d_pruned =  nn.Sequential(*list(resnet18.children())[2:-4])
#res3d_pruned = nn.Sequential(*(list(model.children())[:5]))#+(list(inter1.children()))))#+list(resnet18.children())[2:-4]))
#inter1 = Inter_Layers()


#print(resnet_hybrid(torch.rand([1,3,5,100,100])).shape)
#torch.save(resnet_hybrid,'/home/sebastian/code/deeplabchop-3d/deeplabcut3d/deeplabchop/resnet_hybrid.pth',pickle_protocol=pickle.HIGHEST_PROTOCOL)

data_path='/home/sebastian/Desktop/'
batch_size=3
path_val = None
t_size = 10

path_X = os.path.join(data_path,'ExtractedFrames.h5')
path_Y = os.path.join(data_path,'labels.pkl')

data = h5py.File(path_X,'r')
    
X = data['X']

with open(path_Y, 'rb') as handler:
    Y = pickle.load(handler)

n_categories = len(Y[0].keys())

resnet_hybrid = create_model(5)

criterion = nn.MSELoss()
#    criterion = criterion.cuda()
optimizer = optim.SGD(
    resnet_hybrid.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
    nesterov=nesterov)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=20)

opt = type('', (), {})() #create empty object
opt.arch = 'resnet-101'
opt.result_path = data_path
opt.no_cuda=True
opt.checkpoint=1
train_generator =  generator.Generator(X, Y, batch_size, n_frames = 5, size_x = 400, size_y = 400)
resnet_hybrid.cuda()

print('run')
for i in range(0,100):# n_epochs + 1):
    x,y = train_generator.__getitem__()
    
    y_pred = resnet_hybrid(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(i, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


