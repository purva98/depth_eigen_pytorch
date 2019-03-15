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


class ScaleInvariantLoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantLoss, self).__init__()

    def forward(self, pred, target):
        
        # mask out zero values and invalid regions
        mask = (target == 0) | (target == target.max()) | (target == target.min())
        dist = pred[~mask] - torch.log(target[~mask])
        
        # the lambda parameter is set to 0.5
        loss = torch.mean(dist ** 2) - 0.5 / ((torch.numel(dist)) ** 2) * (torch.sum(dist) ** 2) 
        
        return loss