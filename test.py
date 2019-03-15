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
import shutil
from torchvision.models import vgg16

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from dataset import DepthEigenDataset
from network import GlobalCoarseNet, LocalFineNet
from loss import ScaleInvariantLoss
import util

data_dir_test = Path('nyu/test')
bs = 32
dataloader_test, datalen_test = util.load_test(data_dir_test, bs)
print(datalen_test)

global_model = torch.load('models/global_model.pt')
global_model.eval()

local_model = torch.load('models/local_model.pt')
local_model.eval()

for i, samples in enumerate(dataloader_test):
    rgbs = samples['rgb'].float().to(device)
    depths = samples['depth'].float().to(device)

    # results from global coarse network
    with torch.no_grad():
        global_output = global_model(rgbs).unsqueeze(1)
        #global_output = torch.exp(global_output)
    
    # results from local fine network
    local_model.eval()
    with torch.no_grad():
        local_output = local_model(rgbs, global_output)
    #break

for i in range(2):
    dg = global_output[i].view(74, 55)
    dl = local_output[i].view(74, 55)

    r = transforms.ToPILImage()(rgbs[i].cpu())
    d_true = depths[i]
    util.plot_samples(r,d_true,dg,dl)
    util.plot_histogram(d_true,dg,dl)
    print(d_true)
    print(dg)
    print(dl)