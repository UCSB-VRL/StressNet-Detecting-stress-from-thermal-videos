import numpy as np
import torch
import os
import sys
import random
import torch.nn as nn
import torch.utils.data as tdata
import h5py
import glob
import scipy.io as sio

class BR_Loader:
	def __init__(self, root):
		self.br_sig_dir = root

	def __call__(self, index):
		print("loading the predicted ISTI signal")
		isti_sig = np.load(self.br_sig_dir[index])

		try:
			if(isti_sig.shape[0] == 0):
				print("Nothing to read...")
				return -1
			else:
				return isti_sig
		except:
			if(isti_sig ==  -1): return -1
			else : import pdb; pdb.set_trace()
		
#------------------------------------------------------------------------------------------------------------------------------------#

class thermaldataset(tdata.Dataset):

	def __init__(self, isti_sig_path, phase):
		'''isti_sig sample file name'''# sj160_se03_cv_LJ.mea.mat
		self.isti_sig = isti_sig_path
		self.isti_sig_files = os.listdir(self.isti_sig)
		self.all_dir=[os.path.join(self.isti_sig,a) for a in self.isti_sig_files]
		'''isti signal'''
		self.brloader = BR_Loader(self.all_dir)
		'''train/validate/test phase'''
		self.phase = phase
	
	def __getitem__(self, index):
		#isti data
		br_sig = self.brloader(index)
		#vid filename
		label_fname = self.all_dir[index].split('/')[-1].split('.')[0]
		ses = label_fname.split('_')[1].split('se')[-1].split('0')[-1]

		try:
			if(br_sig.all() == -1): return -1
		except:
			if(br_sig == -1): return -1
		
		#stress classfication label
		if(int(ses) == 2): cls_label = 0
		elif(int(ses) == 3): cls_label = 1

		data_sample = {'cls_label' : cls_label, 'isti_sig' : br_sig }
		return data_sample

	def __len__(self):
		return len(self.all_dir)

#------------------------------------------------------------------------------------------------------------------------------------#
#file usage : python dataloader.py ../data/isti_data 

if __name__=='__main__':
	label_name	  = sys.argv[1]

	print(label_name)
	label	 = "{}/".format(label_name)

	train_dataset = thermaldataset(
		label	 = "{}/".format(label_name), 
		phase='train'
	)
	trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=1)
	for i in trainloader:
		pass
		#import pdb;pdb.set_trace()




