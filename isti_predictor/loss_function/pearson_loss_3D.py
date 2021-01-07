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

class Pearson_Correlation:
	def __init__(self):
		print("Pearson_Correlation Cost function")

	def pearson_correlation(self, x, y, eps=1e-8):
		'''
		Arguments
		---------
		x1 : 3D torch.Tensor
		x2 : 3D torch.Tensor
		batch dim first
		'''
		#import pdb; pdb.set_trace()
		mean_x = torch.mean(x, dim = 2, keepdim=True)
		mean_y = torch.mean(y, dim = 2, keepdim=True)
		xm = x - mean_x
		ym = y - mean_y
		#dot product
		r_num = torch.sum(torch.mul(xm,ym), dim=2, keepdim=True)
		r_den = torch.norm(xm, 2, dim=2, keepdim=True) * torch.norm(ym, 2, dim=2, keepdim=True)
		r_den[torch.where(r_den==0)] = 1.000 # avoid divide by zero
		r_val = r_num / r_den
		
		return r_val

if __name__ == '__main__':
	PC = Pearson_Correlation()

	#import pdb; pdb.set_trace()
	x1 = Variable(torch.randn(1,10,3), requires_grad=True)
	x2 = Variable(torch.randn(1,10,3), requires_grad=True)
	print(x1.shape)
	pc_coeff = PC.pearson_correlation(x1,x2)
