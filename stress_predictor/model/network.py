#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 2 13:11:19 2020

@author: satish
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
print("PyTorch Version:", torch.__version__)

class Attention(nn.Module):
	def __init__(self, br_sig, scale=0.5):
		super().__init__()
		self.representation_size = int(br_sig*scale)

		self.attn = nn.Linear(br_sig, self.representation_size)
	
	def forward(self, x):
		x = torch.squeeze(x)
		x = F.relu(self.attn(x))

		return x

class Classifier(nn.Module):
	def __init__(self, pred_isti, scale=0.5):
		super().__init__()
		self.representation_size = int(pred_isti*scale)

		#self.Attn = Attention(pred_isti)
		self.fc1  = nn.Linear(pred_isti, self.representation_size)
		self.fc2  = nn.Linear(self.representation_size, 1)

	def forward(self, x1):
		x1 = torch.squeeze(x1)
		x1 = F.relu(self.fc1(x1))
		x1 = self.fc2(x1)

		return x1

if __name__ == '__main__':

	classifier = Classifier(512).cuda()
	print(classifier)
	#inputs = torch.rand(1,499,3,240,200).float().cuda()
	#import pdb; pdb.set_trace()
	#out = lstm(inputs)
		
