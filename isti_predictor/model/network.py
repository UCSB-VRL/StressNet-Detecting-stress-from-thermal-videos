#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 2 13:11:19 2020

@author: satish
"""
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torchvision
from torchvision import models
from model.pep_detector import pep_detector
print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

print("Loading ResNet Model")
start_time = time.time()
resnet_model = torchvision.models.resnet50(pretrained=True, progress=True).float()

class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		#loading blocks of ResNet
		blocks     = list(resnet_model.children())[0:8]
		self.convs = nn.Sequential(*blocks)	
		self.avg_p = nn.AdaptiveAvgPool2d(output_size=(1, 1))

	def forward(self, x):
		frames = [frame for frame in x]
		x = self.convs(torch.cat(frames))
		x = self.avg_p(x)
		x = x.view(x.shape[0], -1)
		#x.shape is frames x flat_feature_vector
		
		return x

class Merge_LSTM(nn.Module):
	def __init__(self, in_dim, h_dim, num_l, frame_rate):
		super().__init__()
		self.in_dim 	= in_dim
		self.h_dim  	= h_dim
		self.num_l  	= num_l
		self.frame_rate = frame_rate
		self.cnn    	  = CNN() #initialize CNN
		self.lstm_layer   = nn.LSTM(self.in_dim, self.h_dim, self.num_l, batch_first=True)
		self.detected_pep = pep_detector(30, 4) #initialize linear layers

	def forward(self, x):
		batch_size, timesteps, C, H, W = x.size()
		x = self.cnn(x)
		#timestamp/15 as frame rate is 15 fps. we will push 1 second info to lstm as 1 seq
		x = x.view(batch_size, timesteps//self.frame_rate, -1)
		x_out, (h_o, c_o) = self.lstm_layer(x)
		x_out = x_out[-1].view(batch_size, timesteps, -1).squeeze()
		x_out = self.detected_pep(x_out)
		
		return x_out
	
if __name__ == '__main__':

	lstm = Merge_LSTM(256, 6, 3).cuda()
	print(lstm)
	inputs = torch.rand(1,499,3,240,200).float().cuda()
	#import pdb; pdb.set_trace()
	out = lstm(inputs)

		
