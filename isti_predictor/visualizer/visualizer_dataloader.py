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

sys.path.append(".")
from preprocess_face import preprocess_face, non_linearity


class VideoLoader:
	def __init__(self, root):
		self.vid_dir = root
		self.prev_frame = np.zeros((640,240), dtype=np.float32)

	def __call__(self, vid_fname):
		#pf = preprocess_face()
		print("reading video")
		self.frames = []
		self.par_vid_path = f'{self.vid_dir}/{vid_fname}'
		self.vid_path = glob.glob(self.par_vid_path)
		if(len(self.vid_path)) == 0: 
			print("Video file ", self.par_vid_path, "not found")
			return -1
		
		video_mat = h5py.File(self.vid_path[0],'r')['data']
		print(video_mat)
		if(video_mat.shape[0] == 0):
			print("no frames read...")
		else:
			for idx in range(video_mat.shape[0]-300):
				#print(idx)
				try:
					image  = video_mat[idx,:,:]
				except:
					print("Failed to read frame")
					return self.prev_frame
				image = np.flipud(image)
				image = np.fliplr(image)
				image = np.uint8(image*255)
				self.frames.append(image)
			self.frames = np.array(self.frames)
			
		return self.frames

class LabelLoader:
	def __init__(self, root):
		self.label_dir = root

	def __call__(self,index):
		print("loading ecg signal")
		self.ecg_signal = np.load(self.label_dir[index])

		return self.ecg_signal

class VidEcg_sync:
	def __init__(self, root):
		self.sync_sig_dir = root
		print(self.sync_sig_dir)
		self.ts_file = sio.loadmat(self.sync_sig_dir)['timingCell']

	def __call__(self, sub, ses, task):
		'''each session is a 195 seconds window
		+ve ST(Start Time) means IR video recording started earlier then ECG recodring
		--clip the IR video to t_th second & ECG from zero reading
		-ve ST means IR video recodring started later then ECG recording
		--clip the ECG signal by t seconds & IR video from zero reading
		IR video frame rate : 15hz & ECG signal rate : 1920hz
		1 IR video frame = 128 ECG readings'''
		#sub = 135; ses = 3; task = 'cf'
		#print(sub, ses, task)
		'''type-casting to supress the python FutureWarning'''
		sub_loc  = np.where(self.ts_file[:,0:1] == int(sub))
		ses_loc  = np.where(self.ts_file[sub_loc[0],1:2]== int(ses))
		task_loc = np.where(self.ts_file[sub_loc[0],:][ses_loc[0],2:3] == str(task))
		start_ts = self.ts_file[sub_loc[0],:][ses_loc[0],:][task_loc[0],:][0][3][0][0]
		feetin_ts= self.ts_file[sub_loc[0],:][ses_loc[0],:][task_loc[0],:][0][3][0][2]
		print("start time", start_ts, "Immerse feet: ", feetin_ts)

		return start_ts, feetin_ts

#------------------------------------------------------------------------------------------------------------------------------------#

class thermaldataset(tdata.Dataset):

	def __init__(self, label, ir_video, sync_sig, phase):
		'''ecg label data path'''# sj160_se03_cv_LJ.mea.mat
		self.label = label
		self.label_files = os.listdir(self.label)
		self.all_dir=[os.path.join(self.label,a) for a in self.label_files]
		#Init load label class
		self.labelloader = LabelLoader(self.all_dir)
		'''ir_video data path'''
		self.ir_video = ir_video
		self.videoloader = VideoLoader(self.ir_video)
		'''sync signal for ecg/ir_video'''
		self.sync_sig = sync_sig
		self.sync_sig_file = os.path.join(self.sync_sig, os.listdir(self.sync_sig)[0])
		self.videcg_sync = VidEcg_sync(self.sync_sig_file)
		'''train/validate/test phase'''
		self.phase = phase
    
	def __getitem__(self, index):
		#label data
		label_data = self.labelloader(index)
		#vid filename
		label_fname = self.all_dir[index].split('/')[-1].split('_')
		sub = label_fname[0].split('sj')[-1]
		ses = label_fname[1].split('se')[-1].split('0')[-1]
		vid_fname = f'BOSS_-BOSS_{sub}_1_{ses}_{label_fname[2]}-*'
		#IR video data
		ir_video_data = self.videoloader(vid_fname)
		#In case the video read fails
		try:
			if(ir_video_data.all() == -1 ): 
				print("Error..")
				return -1
		except:
			if(ir_video_data == -1 ): 
				print("File not found")
				return -1

		print("shape of video data", ir_video_data.shape, vid_fname)
		#update corresponding label if person moves out of frame
		#ir video & ecg signal sync
		start_ts, feetin_ts = self.videcg_sync(sub, ses, label_fname[2])
		'''IR vid frame rate : 15hz & ECG rate : 1920hz, 128 label per video frame'''
		if(start_ts > 0):
			start_frame  = math.ceil(start_ts*15)
			if(start_frame > ir_video_data.shape[0]*.75):
				print("getitem, start_frame > ir_vid")
				return -1
			ir_video_data = ir_video_data[start_frame:,:,:]
			label_data = label_data[:(ir_video_data.shape[0]*128)]
			feetin_frame = math.ceil(feetin_ts*15) - start_frame 
		else:
			start_ecg = math.ceil(abs(start_ts)*1920)
			if(start_ecg > label_data.shape[0]*0.75):
				print("getitem, start_ecg> label shape")
				return -1
			label_data = label_data[start_ecg:(start_ecg + (ir_video_data.shape[0]*128))]
			feetin_frame = math.ceil(feetin_ts*15) 
        
		#making length of label_data = len(vid)*128
		rem = label_data.shape[0]%128
		label_pad_len = 0
		if(rem != 0):
			div = label_data.shape[0]/128
			label_pad_len = ((int(div)+1)*128) - label_data.shape[0]
		
		label_data = np.pad(label_data, (0, int(label_pad_len)), 'symmetric').reshape((-1, 128))
		#update corresponding label if person moves out of frame
		print(ir_video_data.shape, label_data.shape)
		#added if the person moves out of the frame then make label 0
		if(ir_video_data.shape[0] != label_data.shape[0]): 
			print("Label and video length mismatch")
			return -1
		data_sample = {'data'     : ir_video_data, 
					   'label'    : label_data, 
					   'feetin_frame' : feetin_frame,
					   'feetin_ts': feetin_ts,
					   'filename' : vid_fname}
		return data_sample

	def __len__(self):
		return len(self.all_dir)

#------------------------------------------------------------------------------------------------------------------------------------#
#file usage : python dataloader.py ../data/test_label ../data/mat_files ../data/sync_data 

if __name__=='__main__':
	label_name    = sys.argv[1]
	ir_vid_name   = sys.argv[2]
	sync_sig_name = sys.argv[3]

	print(label_name, ir_vid_name)
	label    = "{}/".format(label_name)
	ir_video = "{}/".format(ir_vid_name)
	print(label, ir_video)

	train_dataset = thermaldataset(
		label    = "{}/".format(label_name), 
		ir_video = "{}/".format(ir_vid_name), 
		sync_sig = "{}/".format(sync_sig_name), 
        phase='train'
	)
	
	#for i in range(len(train_dataset)):
	#	sample = train_dataset[i]
	#	print(i, sample['video_frames'].shape, sample['label'].shape)
	
	trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=2)
	for i in trainloader:
		pass
		#import pdb;pdb.set_trace()




