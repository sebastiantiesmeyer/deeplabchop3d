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
from torchvision import models as tmodels
import models.resnet as local_models


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


model = local_models.resnet18(sample_size = None,
                         sample_duration = 5,#file['X'].shape[1], 
                         num_classes = file['Y'].shape[-1],
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

#state_dict['out.0.weight']=torch.rand([5,2048,10,1,1])

model_dict.update(state_dict)
model.load_state_dict(model_dict) 
##model=model.cuda()
 
class Inter_Layers(nn.Module):
    def __init__(self):
        super(Inter_Layers, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        
    def forward(self, x):
        x = x.max(dim = 2, keepdim=False)[0]
#        print(x.shape)
#        x = nn.functional.relu(self.conv1(x))
        return x


#model1 =  nn.Sequential(*list(model.children())[:5])

resnet18 = tmodels.resnet18(pretrained=True, )
#res2d_pruned =  nn.Sequential(*list(resnet18.children())[2:-4])
#res3d_pruned = nn.Sequential(*(list(model.children())[:5]))#+(list(inter1.children()))))#+list(resnet18.children())[2:-4]))
#inter1 = Inter_Layers()

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

resnet_hybrid = Resnet_Hybrid(5)

print(resnet_hybrid(torch.rand([1,3,5,100,100])).shape)
torch.save(resnet_hybrid,'/home/sebastian/code/deeplabchop-3d/deeplabcut3d/deeplabchop/resnet_hybrid.pth',pickle_protocol=pickle.HIGHEST_PROTOCOL)

data_path='/home/sebastian/Desktop/'
batch_size=4
path_val = None
t_size = 10

path_X = os.path.join(data_path,'ExtractedFrames.h5')
path_Y = os.path.join(data_path,'labels.pkl')

data = h5py.File(path_X,'r')
    
X = data['X']

with open(path_Y, 'rb') as handler:
    Y = pickle.load(handler)

n_categories = len(Y[0].keys())

criterion = nn.MSELoss()
#    criterion = criterion.cuda()
optimizer = optim.SGD(
    model.parameters(),
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
train_generator =  generator.Generator(X, Y, batch_size, n_frames = 7, size_x = 350, size_y = 350)
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


