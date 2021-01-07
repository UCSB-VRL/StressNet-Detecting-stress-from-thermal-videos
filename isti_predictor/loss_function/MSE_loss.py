#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 6 11:34:06 2020

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

class MSE_loss:
	def __init__(self):
		print("Initializing MSE loss")

	def mse_weighted(self, x, y, mask):
		'''
		Arguments
		---------
		x : target label
		y : prediction/input
		we want to penalize the error more if nearing to the peak
		'''
		self.target   = x
		self.pred     = y
		self.mask     = mask
		
		penalty_pos   = torch.where(self.mask>0.0)
		if(penalty_pos[0].shape[0] == 0): penalty_ratio = 1.0
		else:
			penalty_ratio = (self.target.numel()-penalty_pos[0].shape[0])/penalty_pos[0].shape[0]
		mask[penalty_pos] = mask[penalty_pos]*penalty_ratio
		
		#import pdb; pdb.set_trace()
		sq_error  = torch.sum(((self.pred - self.target) ** 2)*mask)
		mean_loss = sq_error/(self.target.shape[1]*self.target.shape[2])

		return mean_loss

if __name__ == '__main__':
    MSE = MSE_loss()

    #import pdb; pdb.set_trace()
    x1 = Variable(torch.randn(1,10,3), requires_grad=True)
    x2 = Variable(torch.randn(1,10,3), requires_grad=True)
    print(x1.shape)
    mse = MSE.mse_weighted(x1,x2)
