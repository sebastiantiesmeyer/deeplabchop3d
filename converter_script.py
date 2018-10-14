#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:50:21 2018

@author: sebastian
"""

import torch
import numpy as np
from keras.models import load_model

#input_np = Variable(torch.randn([1, 3, 10, 10, 10]))

#torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
model = torch.load('/home/sebastian/code/3D-ResNets-PyTorch/resnet_50_weights.pth')
#keras_output = '/home/sebastian/code/model.hdf5'
#onnx.convert(pytorch_model, keras_output)
#model = load_model(keras_output)
#preds = model.predict(x)


