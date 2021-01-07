#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:11:19 2020

@author: satish
"""

import numpy as np
import cv2
import h5py
import math
from scipy.io import loadmat
import scipy.ndimage as ndimage
from scipy import fftpack
from skimage import util
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable, gradcheck

class FFT_MSE:
	def __init__(self):
		print("Intializing fft and wasserstein loss")
		self.fft_loss = nn.MSELoss()

	def fft_mse(self, x, y, mask):
		'''
		Arguments
		-------------------
		x : 3D torch.Tensor, target
		y : 3D torch.Tensor, prediction
		batch dim first
		'''
		self.x = x
		self.y = y
		self.mask = mask
		#import pdb; pdb.set_trace()
		#retaining the dimension batch_size , legth of whole ecg for all frames(num_frames*num_ecg)
		flat_x = torch.reshape(x,(1, -1))
		flat_y = torch.reshape(y, (1, -1))
		mask   = torch.reshape(mask, (1, -1))
		'''
		#creating mask for the non-important values
		penalty_pos   = torch.where(mask>0.0)
		if(penalty_pos[0].shape[0] == 0): penalty_ratio = 1.0
		else:
			penalty_ratio = (self.flat_x.numel()-penalty_pos[0].shape[0])/penalty_pos[0].shape[0]
		mask[penalty_pos] = mask[penalty_pos]*penalty_ratio
		'''
		#penalty_mask = 1.0
		flat_x = flat_x
		flat_y = flat_y

		#compute fft of both x & y
		fft_x = torch.rfft(flat_x, 1)
		fft_y = torch.rfft(flat_y, 1)

		#magnitue
		mag_x = torch.sqrt(fft_x[:,:,0:1]**2 + fft_x[:,:,1:2]**2)
		mag_y = torch.sqrt(fft_y[:,:,0:1]**2 + fft_y[:,:,1:2]**2)

		#compute phase
		phase_x = torch.atan2(fft_x[:,:,1:2], fft_x[:,:,0:1])
		phase_y = torch.atan2(fft_y[:,:,1:2], fft_y[:,:,0:1])
		
		#compute magnitue MSE loss
		mag_loss = self.fft_loss(mag_x, mag_y)
		#compute phase MSE loss
		phase_loss = self.fft_loss(phase_x, phase_y)

		#print(mag_loss, phase_loss)
		total_fft_loss = (mag_loss + phase_loss)/2

		return total_fft_loss

if __name__ == '__main__':
	PC = FFT_MSE()

	#import pdb; pdb.set_trace()
	x1 = Variable(torch.randn(1,500,128), requires_grad=True)
	x2 = Variable(torch.randn(1,500,128), requires_grad=True)
	print(x1.shape)
	pc_coeff = PC.fft_mse(x1,x2)
	print(pc_coeff)


