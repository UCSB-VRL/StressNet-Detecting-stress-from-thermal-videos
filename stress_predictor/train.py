#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 5 11:00:22 2020

@author: satish
"""
import numpy as np
import os
import sys
import argparse
import random
import glob
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import average_precision_score

from model.network import Classifier as classifier
from dataloader.dataloader import thermaldataset
from loss_function.utils_loss import stress_loss

print("PyTorch Version:", torch.__version__)

#Result Directory
try:
	os.mkdir('../results/')
except OSError as exc:
	pass

def main():
	#Input arguments/training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('-e','--epochs',type=int,required=False,default=100, help='Number_of_Epochs')
	parser.add_argument('-lr','--learning_rate',type=float,required=False,default=0.01, \
						help='Learning_Rate')
	parser.add_argument('-ba','--batch_size',type=int,required=False, default=1,help='Batch_Size')
	parser.add_argument('-seed',type=int,required=False, default=5,help='random seed')
	parser.add_argument('-data',type=str,required=False, default='../data/normalized_pep_labels/', \
						help='data path')
	parser.add_argument('-phase',type=str,required=False, default='train',help='train/test mode')
	parser.add_argument('-split','--train_val_split', type=float, required=False, default=0.8, \
						help='train/test mode')
	parser.add_argument('-min_batch', '--frames_in_GPU',type=int,required=False, default=120, \
						help='number of frames per batch from the video to go in GPU')
	parser.add_argument('-pf', '--pep_frames',type=int,required=False, default=150, \
						help='number of pep values to classify stress type')

	#Parameters for existing model reload
	parser.add_argument('-resume',type=str,required=False, default='F',help='resume training')
	parser.add_argument('-hyper_param',type=str,required=False, default='F', \
						help='existing hyper-parameters')
	parser.add_argument('-cp','--checkpoint_path',type=str,required=False, \
						default='../model_checkpoints_gt_isti_cls/', help='resume training')
	
	args   = parser.parse_args()
	epochs = args.epochs
	l_rate = args.learning_rate
	data   = args.data
	phase  = args.phase
	seed   = args.seed
	split  = args.train_val_split
	fps    = args.frames_in_GPU  #numbers of frames per batch
	pf	   = args.pep_frames #number of pep values to classify stress type
	batch_size = args.batch_size

	#Parameters for exisitng model reload
	c_point_dir = args.checkpoint_path
	resume		= args.resume
	hyper_param = args.hyper_param
	cur_epoch	= 0

	#Fix seed
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed) #sets the seed for random number
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)

	#Initializing Network
	print("Initializing Network")
	cls_model = classifier(pf).cuda()

	#optimizer
	optimizer = optim.Adam(cls_model.parameters(), lr=l_rate)
	
	lambda1   = lambda epoch : 1.0 if epoch<10 else (0.1 if epoch<20 else 0.001)
	scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

	#Dataloader
	print("Initializing dataloader")
	datasets  = {}
	dataset   = thermaldataset(data, phase='train')
	'''Indexes for train/val'''
	idxs = list(range(0, len(dataset)))
	random.shuffle(idxs)
	split_idx = int(len(idxs) * split)
	trainIdxs = idxs[:split_idx]; valIdxs = idxs[split_idx:]
	'''create subsets'''
	datasets['train'] = torch.utils.data.Subset(dataset, trainIdxs)
	datasets['test']  = torch.utils.data.Subset(dataset, valIdxs)
	print(len(datasets['train']))
	print(datasets['train'].dataset)
	dataloader_tr  = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, \
												shuffle=True, num_workers=0)
	dataloader_val = torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, \
												shuffle=True, num_workers=0)
	dataloader = {'train': dataloader_tr, 'test' : dataloader_val}

	#Loss function
	print("Initializing loss function")
	cls_loss = stress_loss()

	#Load the existing Model
	if resume == 'T':
		try:
			checkpoint_dir = f'{c_point_dir}/*'
			f_list = glob.glob(checkpoint_dir)
			latest_checkpoint = max(f_list, key=os.path.getctime)
			print("Resuming Existing state of Pretrained Model")
			checkpoint = torch.load(latest_checkpoint)
			cls_model.load_state_dict(checkpoint['model_state_dict'])
			cur_epoch  = checkpoint['epoch']
			cur_loss  = checkpoint['loss']
			print("Loading Done, Loss: {}, Current_epoch: {}".format(cur_loss, cur_epoch))
		except:
			print("Loading Existing state of Pretrained Model Failed ! ")
			print("Initializing training from epoch 0, PRESS c to continue")
			import pdb; pdb.set_trace()

	#Loading the existing Hyperparameters
	if hyper_param == 'T':
		try:
			checkpoint_dir = f'{c_point_dir}/*'
			f_list = glob.glob(checkpoint_dir)
			latest_checkpoint = max(f_list, key=os.path.getctime)
			checkpoint = torch.load(latest_checkpoint)
			print("Loading existing hyper-parameters")
			optimizer.load_state_dict(checkpoint['optimizer'])
			scheduler.load_state_dict(checkpoint['scheduler'])
		except:
			print("Failed to load existing hyper-parameters")

	#Initialization done, Start training loop
	#parameters
	params = {'cur_epoch':cur_epoch,
				'epochs' :epochs,
				'phase'  :phase,
				'fps'	 :fps,
				'data'	 :data,
				'batch'  :batch_size,
				'pf'	 :pf}

	training_loop(cls_model, optimizer, scheduler, dataloader, cls_loss, **params)

#Save Checkpoint
def save_checkpoint(state, filename):
	checkpoint_path = f'../model_checkpoints_gt_isti_cls/{filename}'
	torch.save(state, checkpoint_path)
	return
	
def training_loop(cls_model, optimizer, scheduler, dataloader, cls_loss, **params):
	#training vars
	best_train_loss  = 100
	mean_train_loss  = 100
	best_epoch_train = 0
	#test vars
	best_test_loss	 = 100	
	mean_test_loss	 = 100
	best_epoch_test  = params['cur_epoch']

	plot_train_loss = []
	plot_test_loss	= []
	plot_train_acc	= []
	plot_test_acc	= []

	train_predicted_cls   = torch.zeros((0,)).cuda().float()
	train_target_cls	  = torch.zeros((0,)).cuda().float()
	
	test_predicted_cls	 = torch.zeros((0,)).cuda().float()
	test_target_cls		 = torch.zeros((0,)).cuda().float()
	
	for epoch in range(params['cur_epoch'], params['epochs']):
		optimizer.zero_grad()
		scheduler.step()
		train_loss	 = 0.0
		train_acc	 = 0.0
		print('Epoch {}/{}'.format(epoch, params['epochs']-1))

		if params['phase'] == 'train':
			cls_model.train()	#set model to training mode
			#Iterate over data
			for iteration, data in enumerate(tqdm(dataloader['train'])):
				running_loss = 0.0
				running_acc  = 0.0
				try:
					cls_label = data['cls_label'].cuda().float()
					isti_sig = data['isti_sig'].cuda().float()

				except:
					print("Data read error")
					continue

				print("Stats ", cls_label, isti_sig.shape, iteration)
				
				if(isti_sig.shape[1]<150): 
					print("Size too small, skipping it")
					continue
				
				#forward, track history if only in train
				with torch.set_grad_enabled(params['phase'] == 'train'):	
					#classifier model
					pred_cls = cls_model(isti_sig[:,0:params['pf']])
					train_predicted_cls = torch.cat((train_predicted_cls, pred_cls), dim=0)
					train_target_cls	= torch.cat((train_target_cls, cls_label), dim=0)

					classification_loss = cls_loss(pred_cls, cls_label)
					classification_loss.backward()

					running_loss = running_loss + classification_loss.item()
					train_loss	 = train_loss +  running_loss

					if (iteration+1)%2 == 0:
						optimizer.step()
						optimizer.zero_grad()

				#mean training loss
				mean_train_loss = train_loss/(iteration+1)

				cur_training_vars = {'Training_loss': mean_train_loss,
									 'Phase'		: params['phase'],
									 'epoch'		: epoch+1,
									 'iteration'	: iteration+1,
									 'Leaning Rate' : scheduler.get_lr()
									}
				best_training_vars= {'Best_train_loss': best_train_loss,
									 'Min Loss epoch' : best_epoch_train
									}

				print("Current Training Vars: ", cur_training_vars, "Best Training Vars: "\
																	, best_training_vars)
			try:
				mean_train_loss = mean_train_loss.item()
			except:
				mean_train_loss = mean_train_loss
			plot_train_loss.append(mean_train_loss)
			if mean_train_loss < best_train_loss:
				best_train_loss  = mean_train_loss
				best_epoch_train = epoch+1
		
		m = nn.Sigmoid()
		AP = average_precision_score(train_target_cls.cpu().detach().numpy(), m(train_predicted_cls).cpu().detach().numpy())

		print("***********Train AP**************", AP)

		#Validating the model
		test_loss = 0
		test_acc = 0
		for iteration, data in enumerate(tqdm(dataloader['test'])):
			running_loss   = 0.0
			running_acc    = 0.0
			cls_model.eval()

			try:
				cls_label = data['cls_label'].cuda().float()
				isti_sig  = data['isti_sig'].cuda().float()
			except:
				print("Data read error, corrupted data")
				continue

			if(isti_sig.shape[1]<150):
				print("Size too small, skipping it")
				continue

			print("label: ", cls_label, isti_sig.shape, iteration)
			with torch.no_grad():
				pred_cls = cls_model(isti_sig[:, 0:params['pf']])
				test_predicted_cls = torch.cat((test_predicted_cls, pred_cls), dim=0)
				test_target_cls    = torch.cat((test_target_cls, cls_label), dim=0)

				classification_loss = cls_loss(pred_cls, cls_label)	
				running_loss= running_loss + classification_loss
				test_loss	= test_loss + running_loss

			#mean test loss and acc
			mean_test_loss = test_loss/(iteration+1)

			cur_test_vars = {'Test_loss': mean_test_loss,
							 'Phase'	: 'Test',
							 'epoch'	: epoch+1,
							 'iteration': iteration+1
							}
			best_test_vars= {'Best_test_loss':best_test_loss,
							 'Min Loss epoch':best_epoch_test
							}
			print("Current Test Vars: ", cur_test_vars, "Best Test Vars: ", best_test_vars)

		AP = average_precision_score(test_target_cls.cpu().detach().numpy(), m(test_predicted_cls).cpu().detach().numpy())

		print("***********Test AP**************", AP)

		try:
			mean_test_loss = mean_test_loss.item()
		except:
			mean_test_loss = mean_test_loss

		plot_test_loss.append(mean_test_loss)
		if mean_test_loss < best_test_loss:
			best_test_loss	= mean_test_loss
			best_epoch_test = epoch+1
			time_stamp		 = datetime.now().strftime("%Y_%m_%d_%H_%M_%s")

			save_checkpoint({'model_state_dict': cls_model.state_dict(),
							 'epoch'		   : epoch + 1,
							 'loss'			   : best_test_loss,
							 'optimizer'	   : optimizer.state_dict(),
							 'scheduler'	   : scheduler.state_dict()}, 
							 'checkpoint_{0}.pth.tar'.format(time_stamp))

	#saving loss for train and test
	loss_dump = {"train" : plot_train_loss,
				 "test"  : plot_test_loss,
				 "test_acc" : plot_test_acc}

	with open('../train_test_loss/loss_dump.json', 'w') as json_file:
		json.dump(loss_dump, json_file)

if __name__ == '__main__':
	main()
