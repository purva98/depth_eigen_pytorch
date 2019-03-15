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

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

depths_mean = [2.53434899]
depths_std = [1.22576694]
#depths_istd = 1.0 / depths_std

logdepths_mean = 0.82473954
logdepths_std = 0.45723134
logdepths_istd = 1.0 / logdepths_std

# NYU dataset statistics
images_mean = [109.31410628 / 255, 109.31410628 / 255, 109.31410628 / 255]
images_std = [76.18328376 / 255, 76.18328376 / 255, 76.18328376 / 255]
#images_istd = 1.0 / images_std


class DepthEigenDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        super(DepthEigenDataset, self).__init__()
        
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(self.rgb_dir))
    
    def __getitem__(self, idx):
         
        # read as PIL images
        im_array = np.load(self.rgb_dir / '{}.npy'.format(idx))
        im_array = np.einsum('ijk->kji', im_array)
        depth_array = np.load(self.depth_dir / '{}.npy'.format(idx))
        depth_array = np.einsum('ij->ji', depth_array)
        rgb_sample = Image.fromarray(im_array)
        depth_sample = Image.fromarray(depth_array)
        
        # transform
        seed = random.randint(0, 2 ** 32)
        if self.transform:
            random.seed(seed)
            rgb_sample = self.transform(rgb_sample)
            
            random.seed(seed)
            depth_sample = self.transform(depth_sample)
        
        # resize depth image
        depth_sample = transforms.Resize((74, 55))(depth_sample)
        
        # convert to torch tensor
        rgb_sample = transforms.ToTensor()(rgb_sample)
        rgb_sample =  transforms.Normalize(images_mean, images_std)(rgb_sample)
        depth_sample = transforms.ToTensor()(depth_sample).view(74, 55)
        
        # normalize
        #rgb_sample =  transforms.Normalize(images_mean, images_std)(rgb_sample)
        #depth_sample = transforms.Normalize(depths_mean, depths_std)(depth_sample).view(74, 55)
        
        sample = {'rgb':rgb_sample, 'depth': depth_sample}
        
        return sample