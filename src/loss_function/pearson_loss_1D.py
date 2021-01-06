#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 7 01:01:12 2020

@author: satish
"""
import numpy as np
import os
import sys
import argparse
import random
import torch
import torch.nn as nn
from torch.autograd import Variable, gradcheck

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Pearson_Correlation:
	def __init__(self):
		print("Pearson_Correlation Cost function")

	def pearson_correlation(self, x, y, eps=1e-8):
		'''
		Arguments
		---------
		x1 : 1D torch.Tensor
		x2 : 1D torch.Tensor
		batch dim first
		'''
		x = x.to(device).squeeze()
		y = y.to(device).squeeze()
		mean_x = torch.mean(x)
		mean_y = torch.mean(y)
		xm = x - mean_x
		ym = y - mean_y
		#dot product
		r_num = torch.sum(torch.mul(xm,ym))
		r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
		if(r_den == 0): r_den=1.000
		r_val = r_num / r_den
		
		return r_val

if __name__ == '__main__':
	PC = Pearson_Correlation()

	#import pdb; pdb.set_trace()
	x1 = Variable(torch.randn(10), requires_grad=True)
	x2 = Variable(torch.randn(10), requires_grad=True)
	print(x1.shape)
	pc_coeff = PC.pearson_correlation(x1,x2)
