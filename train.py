from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
from skimage import transform
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import time
import os
import re
import random
from imageio import imread
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

from dataset import DepthEigenDataset
from network import GlobalCoarseNet, LocalFineNet
from loss import ScaleInvariantLoss
import util

#cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
num_epochs = 100 # not specified in paper


data_dir_train = Path('nyu/train')
data_dir_valid = Path('nyu/test')
bs=32
dataloader_train, dataloader_valid, datalen_train, datalen_valid = util.load_train(data_dir_train, data_dir_valid, bs)
print(datalen_train, datalen_valid)

#now the net
# initialize
global_model = GlobalCoarseNet(init=False).to(device)
local_model = LocalFineNet(init=False).to(device)

# loss
global_criterion = ScaleInvariantLoss()
local_criterion = ScaleInvariantLoss()

# optimizer
r = 0.1
global_optimizer = torch.optim.SGD([{'params': global_model.coarse6.parameters(), 'lr': 0.1*r},
                                    {'params': global_model.coarse7.parameters(), 'lr': 0.1*r}], 
                                   lr=0.001*r, momentum=0.9, weight_decay=0.1)

local_optimizer = torch.optim.SGD([{'params': local_model.fine2.parameters(), 'lr': 0.01*r}], 
                                  lr=0.001*r, momentum=0.9, weight_decay=0.1)

# data parallel
global_model = nn.DataParallel(global_model)
local_model = nn.DataParallel(local_model)

#train


train_losses = []
valid_losses = []
tl_b = []

start = time.time()
for epoch in range(num_epochs):
    
    print('>', end=' ')
    
    # train
    train_loss = 0
    global_model.train()
    for i, samples in enumerate(dataloader_train):
        
        rgbs = samples['rgb'].float().to(device)
        depths = samples['depth'].float().to(device)
        
        # forward pass
        output = global_model(rgbs)
        loss = global_criterion(output, depths)

        # backward pass
        global_model.zero_grad()
        loss.backward()

        # optimization
        global_optimizer.step()
        
        train_loss += loss.item()
        tl_b.append(loss.item())
        
    train_losses.append(train_loss / datalen_train)
    
    # validation
    valid_loss = 0
    global_model.eval()
    with torch.no_grad():
        for i, samples in enumerate(dataloader_valid):
        
            rgbs = samples['rgb'].float().to(device)
            depths = samples['depth'].float().to(device)

            # forward pass
            output = global_model(rgbs)
            loss = global_criterion(output, depths)
            
            valid_loss += loss.item()
            
    valid_losses.append(valid_loss / datalen_valid)
    
    # save model
    torch.save(global_model, './models/global_model.pt')
    print(('epoch {epoch} done, train_loss = {tloss}, validation loss = {vloss}').format(epoch=epoch, tloss=train_loss, vloss=valid_loss))

    
elapse = time.time() - start 
print('Time used: ', elapse, ' per epoch used: ', elapse / num_epochs)

train_losses_, valid_losses_ = [], []
tl_b_ = []
start = time.time()
for epoch in range(num_epochs):
    
    print('>', end=' ')
    
    # train
    train_loss = 0
    local_model.train()
    for i, samples in enumerate(dataloader_train):
        
        rgbs = samples['rgb'].float().to(device)
        depths = samples['depth'].float().to(device)
        
        # results from global coarse network
        global_model.eval()
        with torch.no_grad():
            global_output = global_model(rgbs).unsqueeze(1)
            #global_output = global_output * depths_std[0] + depths_mean[0]

        # forward pass
        output = local_model(rgbs, global_output).squeeze(1)
        loss = local_criterion(output, depths)

        # backward pass
        local_model.zero_grad()
        loss.backward()

        # optimization
        local_optimizer.step()
        
        train_loss += loss.item()
        tl_b_.append(loss.item())
        
    train_losses_.append(train_loss / datalen_train)

    # valid
    valid_loss = 0
    local_model.eval()
    with torch.no_grad():
        for i, samples in enumerate(dataloader_valid):

            rgbs = samples['rgb'].float().to(device)
            depths = samples['depth'].float().to(device)

            # results from global coarse network
            global_model.eval()
            with torch.no_grad():
                global_output = global_model(rgbs).unsqueeze(1)

            # forward pass
            output = local_model(rgbs, global_output).squeeze(1)
            loss = local_criterion(output, depths)
            
            valid_loss += loss.item()
    valid_losses_.append(valid_loss / datalen_valid)
    
    # save model
    torch.save(local_model, './models/local_model.pt')
    print(('epoch {epoch} done, train_loss = {tloss}, validation loss = {vloss}').format(epoch=epoch, tloss=train_loss, vloss=valid_loss))

elapse = time.time() - start 
print('Time used: ', elapse, ' per epoch used: ', elapse / num_epochs)

util.plot_losses(train_losses, valid_losses)
util.plot_train_loss(tl_b)
util.plot_losses(train_losses_, valid_losses_)
util.plot_train_loss(tl_b_)

for i, samples in enumerate(dataloader_valid):

    rgbs = samples['rgb'].float().to(device)
    depths = samples['depth'].float().to(device)

    # results from global coarse network
    global_model.eval()
    with torch.no_grad():
        global_output = global_model(rgbs).unsqueeze(1)
        #global_output_ = global_output * depths_std[0] + depths_mean[0]
    
    # results from local fine network
    local_model.eval()
    with torch.no_grad():
        local_output = local_model(rgbs, global_output)
        #local_output = local_output * depths_std[0] + depths_mean[0]
    
    break

for i in range(5):
    dg = global_output[i].view(74, 55)
    dl = local_output[i].view(74, 55)
    
    #r = transforms.Normalize(mean=[], std=[])(rgbs[i])
    #r = rgbs[i] * 76 + 109
    
    r = transforms.ToPILImage()(rgbs[i].cpu())
    
    d_true = depths[i]
    util.plot_samples(r,d_true,dg,dl)