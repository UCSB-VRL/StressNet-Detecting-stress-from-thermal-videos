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
import math
from torchvision import transforms

from utils.preprocess_face import preprocess_face, non_linearity

class VideoLoader(preprocess_face, non_linearity):
	def __init__(self, root):
		self.vid_dir = root
		#spatial&temporal sigma
		self.std_sw=4; self.std_sh=4; self.std_t=3
		preprocess_face.__init__(self)
		non_linearity.__init__(self, self.std_sw, self.std_sh, self.std_t)
		'''
		labels mean, std = 0.03598024, 0.20771082
		'''

	def standardize_frame(self, frame):
		frame = (frame - 0.63839126)/(0.26499870)
		return frame

	def __call__(self, vid_fname):
		print("reading video")
		self.frames = []
		self.label_modifier = []
		self.par_vid_path = f'{self.vid_dir}/{vid_fname}'
		self.vid_path = glob.glob(self.par_vid_path)
		if(len(self.vid_path)) == 0: 
			print("Video file ", self.par_vid_path, "not found")
			return -1, -1
		
		video_mat = h5py.File(self.vid_path[0],'r')['data']
		print(video_mat)
		if(video_mat.shape[0] == 0):
			print("no frames read...")
		else:
			for idx in range(video_mat.shape[0]-300):
				fr, lb = self.face_tube(idx, video_mat)
				fr = self.standardize_frame(fr)
				self.frames.append(fr)
				self.label_modifier.append(lb)
			self.frames = np.array(self.frames)
			self.frames = self.gaus_act(self.frames)
			self.label_modifier = np.array(self.label_modifier)
			
		return self.frames, self.label_modifier[:self.frames.shape[0]]

class LabelLoader:
	def __init__(self, root, labels_per_frame):
		self.label_dir = root
		self.labels_per_frame = labels_per_frame

	def __call__(self,index):
		print("loading isti signal")
		self.isti_signal = np.load(self.label_dir[index])
		#take derivate as we took for video frames too and normalize.
		self.isti_signal -= self.isti_signal.min()
		self.isti_signal /= self.isti_signal.max()
		self.isti_signal = np.diff(self.isti_signal)

		return self.isti_signal

class VidEcg_sync:
	def __init__(self, root):
		self.sync_sig_dir = root
		print(self.sync_sig_dir)
		self.ts_file = sio.loadmat(self.sync_sig_dir)['timingCell']

	def __call__(self, sub, ses, task):
		'''each session is a 195 seconds window
		+ve ST(Start Time) means IR video recording started earlier then ISTI recodring
		--clip the IR video to t_th second & ISTI from zero reading
		-ve ST means IR video recodring started later then ISTI recording
		--clip the ISTI signal by t seconds & IR video from zero reading
		IR video frame rate : 15hz
		1 IR video frame = 30 ISTI readings'''
		#sub = 135; ses = 3; task = 'cf'
		#print(sub, ses, task)
		'''type-casting to supress the python FutureWarning'''
		sub_loc = np.where(self.ts_file[:,0:1] == int(sub))
		ses_loc = np.where(self.ts_file[sub_loc[0],1:2]== int(ses))
		task_loc = np.where(self.ts_file[sub_loc[0],:][ses_loc[0],2:3] == str(task))
		start_ts = self.ts_file[sub_loc[0],:][ses_loc[0],:][task_loc[0],:][0][3][0][0]
		print("start time", start_ts)

		return start_ts
#------------------------------------------------------------------------------------------------------------------------------------#

class thermaldataset(tdata.Dataset):

	def __init__(self, label, ir_video, sync_sig, phase):
		'''isti label data path'''# sj160_se03_cv_LJ.mea.mat
		self.label		 = label
		self.frame_rate  = 15
		self.labels_per_frame = 1
		self.label_files = os.listdir(self.label)
		self.all_dir	 = [os.path.join(self.label,a) for a in self.label_files]	
		#Init load label class
		self.labelloader = LabelLoader(self.all_dir, self.labels_per_frame)
		
		'''ir_video data path'''
		self.ir_video	 = ir_video
		self.videoloader = VideoLoader(self.ir_video)
		
		'''sync signal for ecg/ir_video'''
		#you can skip this if your data is already in sync
		self.sync_sig	   = sync_sig
		self.sync_sig_file = os.path.join(self.sync_sig, os.listdir(self.sync_sig)[0])
		self.videcg_sync   = VidEcg_sync(self.sync_sig_file)
		
		'''train/validate/test phase'''
		self.phase = phase

		'''transforms'''
		self.transform_norm = transforms.Compose([
        		transforms.ToTensor(),
	            transforms.Normalize([0.6384], [0.2650])
    	    ])

	def __getitem__(self, index):
		#label data
		label_data = self.labelloader(index)
		
		#vid filename
		label_fname = self.all_dir[index].split('/')[-1].split('_')
		sub			= label_fname[0].split('sj')[-1]
		ses			= label_fname[1].split('se')[-1].split('0')[-1]
		vid_fname	= f'BOSS_-BOSS_{sub}_1_{ses}_{label_fname[2]}-*'
		#stress label
		stress_label = 0
		if int(ses) == 3: stress_label = 0 #no stress condition
		if int(ses) == 2: stress_label = 1 #stress condition
		#IR video data
		ir_video_data, cur_label = self.videoloader(vid_fname)
		#In case the video read fails
		try:
			if(ir_video_data.all() == -1 or cur_label.all() == -1): return -1
		except:
			if(ir_video_data == -1 or cur_label == -1): return -1
		print("shape of video data and name of file loaded", ir_video_data.shape, vid_fname)
		
		#ir video & ecg signal sync
		start_ts = self.videcg_sync(sub, ses, label_fname[2])
		'''IR vid frame rate : 15hz'''
		'''label fram rate is 120hz: i.e. 8 labels per video frame'''
		if(start_ts > 0):
			start_frame		= start_ts*15
			start_frame_int = math.ceil(start_frame)
			
			# compensate for round-off to integer value
			start_isti = 0
			print("start_frame", start_frame, start_isti)
			
			if(start_frame > ir_video_data.shape[0]*.75):
				print("start_frame > ir_vid : skipping this file")
				return -1
			#if labels were recorded for shorter period of time
			ir_video_data = ir_video_data[start_frame_int:,:,:]
			read_length = min(ir_video_data.shape[0], label_data.shape[0])
			ir_video_data = ir_video_data[:read_length,:,:]
			cur_label	  = cur_label[start_frame_int:start_frame_int+read_length]
			label_data	  = label_data[start_isti:ir_video_data.shape[0]]
		else:
			start_isti	  = math.ceil(abs(start_ts)*15)
			if(start_isti > label_data.shape[0]*0.75):
				print("start_ecg> label shape : skipping this file")
				return -1
			label_data	  = label_data[start_isti:]
			read_length = min(ir_video_data.shape[0], label_data.shape[0])
			ir_video_data = ir_video_data[:read_length]
			cur_label	  = cur_label[:read_length]
			label_data	  = label_data[:read_length]
		
		#making length of label_data = len(vid)*128
		rem = label_data.shape[0]%1
		label_pad_len = 0
		if(rem != 0):
			div = label_data.shape[0]/1
			label_pad_len = ((int(div)+1)*1) - label_data.shape[0]	
		label_data = np.pad(label_data, (0, int(label_pad_len)), 'symmetric').reshape((-1, 1))

		#update corresponding label if person moves out of frame
		cur_label = np.repeat(cur_label, self.labels_per_frame) #to make size equal to label length
		cur_label = cur_label.reshape(-1,1)
		#added if the person moves out of the frame then make label 0
		try:
			label_data = label_data*cur_label
		except:
			print("label and video length mismatch : skipping this file")
			return -1
		
		#making data length multiple of 15
		ir_video_data = ir_video_data[0:(ir_video_data.shape[0]//self.frame_rate)*self.frame_rate]
		#adding channel dimension: b_size, num_frames, h, w, ch
		if len(ir_video_data.shape)<5:
			ir_video_data = np.expand_dims(ir_video_data, axis=1)
		label_data	  = label_data[0:(ir_video_data.shape[0]*self.labels_per_frame)]
		data_sample = {'data': ir_video_data, 'label' : label_data, 's_label' : stress_label}
		
		return data_sample

	def __len__(self):
		return len(self.all_dir)

#------------------------------------------------------------------------------------------------------------------------------------#
#file usage : python dataloader.py ../data/test_label ../data/mat_files ../data/sync_data 
if __name__=='__main__':
	label_name	  = sys.argv[1]
	ir_vid_name   = sys.argv[2]
	sync_sig_name = sys.argv[3]

	print(label_name, ir_vid_name)
	label	 = "{}/".format(label_name)
	ir_video = "{}/".format(ir_vid_name)
	print(label, ir_video)

	train_dataset = thermaldataset(
		label	 = "{}/".format(label_name), 
		ir_video = "{}/".format(ir_vid_name), 
		sync_sig = "{}/".format(sync_sig_name), 
		phase='train'
	)
	
	trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=1)
	for i in trainloader:
		pass
		#import pdb;pdb.set_trace()




