"""
Read data
"""
import h5py
import numpy as np
import random


filepath = 'nyu/nyu_depth_v2_labeled.mat'
f = h5py.File(filepath, 'r')
images = np.array(f['images'])
depths = np.array(f['depths'])
np.random.seed(10)
np.random.shuffle(images)
np.random.seed(10)
np.random.shuffle(depths)
i=0
while i < 1000:
	image = images[i,:]
	depth = depths[i,:]
	np.save("nyu/train/i/"+ str(i) +".npy",image)
	np.save("nyu/train/d/"+ str(i) +".npy",depth)
	i+=1
	print(str(i))
while i < 1200:
	image = images[i,:]
	depth = depths[i,:]
	np.save("nyu/val/i/"+ str(i-1000) +".npy",image)
	np.save("nyu/val/d/"+ str(i-1000) +".npy",depth)
	i+=1
	print(str(i))	
while i < 1449:
	image = images[i,:]
	depth = depths[i,:]
	np.save("nyu/test/i/"+ str(i-1200) +".npy",image)
	np.save("nyu/test/d/"+ str(i-1200) +".npy",depth)
	i+=1
	print(str(i))
