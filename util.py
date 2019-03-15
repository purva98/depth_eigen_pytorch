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
from dataset import DepthEigenDataset

def load_train(data_dir_train, data_dir_valid, bs=32):
	trans_train = transforms.Compose([
	transforms.Resize((320, 240)),
	transforms.RandomRotation(5),
	transforms.RandomCrop((304, 228)),
	transforms.RandomHorizontalFlip(),
	])

	trans_test = transforms.Compose([
		transforms.Resize((304, 228)),
	])
		
	dataset_train = DepthEigenDataset(data_dir_train / 'i', data_dir_train / 'd', transform=trans_train)
	dataloader_train = DataLoader(dataset_train, batch_size=bs, shuffle=True)

	dataset_valid = DepthEigenDataset(data_dir_valid / 'i', data_dir_valid / 'd', transform=trans_test)
	dataloader_valid = DataLoader(dataset_valid, batch_size=bs, shuffle=True)

	datalen_train = len(dataset_train)
	datalen_valid = len(dataset_valid)

	return dataloader_train, dataloader_valid, datalen_train, datalen_valid

def load_test(data_dir_test, bs=32):
	
	trans_test = transforms.Compose([
		transforms.Resize((304, 228)),
	])

	dataset_test = DepthEigenDataset(data_dir_test / 'i', data_dir_test / 'd', transform=trans_test)
	dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

	datalen_test = len(dataset_test)

	return dataloader_test, datalen_test

def compute_errors(gt, pred):
	thresh = np.maximum((gt / pred), (pred / gt))
	a1 = (thresh < 1.25   ).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	rmse = (gt - pred) ** 2
	rmse = np.sqrt(rmse.mean())

	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	abs_rel = np.mean(np.abs(gt - pred) / gt)

	sq_rel = np.mean(((gt - pred)**2) / gt)

	return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def plot_samples(r,d_true,dg,dl):
	figsize(12.5,4)

	plt.subplot(141)
	plt.title('rgb input')
	plt.imshow(r)
	
	plt.subplot(142)
	plt.title('ground truth')
	plt.imshow(d_true.cpu())
	
	plt.subplot(143)
	plt.title('global prediction')
	plt.imshow(torch.exp(dg).cpu())
	
	plt.subplot(144)
	plt.title('local prediction')
	plt.imshow(torch.exp(dl).cpu())
	plt.show()

def plot_histogram(d_true,dg,dl):
	figsize(12.5, 12)
	plt.subplot(321)
	plt.title('ground truth')
	plt.hist(d_true.cpu().view(-1), bins=30)

	plt.subplot(322)
	plt.title('ground truth log')
	plt.hist(torch.log(d_true.cpu().view(-1)), bins=30)

	plt.subplot(323)
	plt.title('global pred')
	plt.hist(torch.exp(dg).cpu().view(-1), bins=30)

	plt.subplot(324)
	plt.title('global pred log')
	plt.hist(dg.cpu().view(-1), bins=30)

	plt.subplot(325)
	plt.title('local pred')
	plt.hist(torch.exp(dl).cpu().view(-1), bins=30)

	plt.subplot(326)
	plt.title('local pred log')
	plt.hist(dl.cpu().view(-1), bins=30)
	plt.show()


#plot loss
def plot_losses(train_losses, valid_losses):
	figsize(12.5, 4)
	plt.plot(train_losses, label='train losses')
	plt.plot(valid_losses, label='valid losses')
	
	plt.xlabel("Iterations")
	plt.ylabel("Losses")
	
	plt.legend()
	plt.title("Losses")
	plt.grid(True)
	plt.show()

def plot_train_loss(tl_b):
	figsize(12.5, 12)

	plt.subplot(311)
	plt.plot(tl_b, label='train loss')
	plt.grid(True)
	plt.legend()

	plt.subplot(312)
	plt.plot(tl_b[100:], label='train loss')
	plt.grid(True)
	plt.legend()

	plt.subplot(313)
	plt.plot(tl_b[300:], label='train loss')
	fml = np.mean(tl_b[-320:])
	plt.axhline(y = fml, color='r', linestyle='-', label='final mean train loss: {:.2f}'.format(fml))
	plt.grid(True)
	plt.legend()

	plt.show()