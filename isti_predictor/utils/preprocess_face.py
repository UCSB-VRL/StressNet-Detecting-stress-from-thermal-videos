#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:11:19 2020

@author: satish
"""
import cv2
import numpy as np
import h5py
import glob
#import dlib
#from imutils import face_utils
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import savgol_filter, argrelextrema, gaussian
from skimage.measure import block_reduce

class preprocess_face:
	def __init__(self):
		print("Initializing class preprocess_face")
		self.prev_frame = np.zeros((360,240), dtype=np.float32)
		self.prev_y = 300

	def face_tube(self, idx, video_mat):
		'''apply OTSU filter on the image and crop the persons face'''
		#h5py read fails sometimes, issue posted in github of h5py
		try:
			image  = video_mat[idx,:,:]
			cur_label = 1
		except:
			print("Failed to read frame")
			cur_label=0
			return self.prev_frame, cur_label
		if(image.shape[0] != 640 or image.shape[1] != 240):
			image = cv2.resize(image, (240, 540), interpolation=cv2.INTER_AREA)
		#print(image.shape)
		image  = np.flipud(image)
		image  = np.fliplr(image)
		org_img = image.copy()
		th_img = np.uint8(image*255)
		ret,th = cv2.threshold(th_img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		image  = image*th
		#pred_img = th_img*th
		'''take derivate of intensity sum to look for face edges'''
		x_sum  = savgol_filter(np.sum(image, axis=0), 31, 3)
		y_sum  = savgol_filter(np.sum(image, axis=1), 51, 3)
		x_edge = argrelextrema(x_sum, np.less, order=50)
		y_edge = argrelextrema(y_sum, np.less, order=40)
		h, w   = image.shape
		'''pad to write crop image into'''
		pad = np.zeros((360,240), dtype=np.float32)
		pad_h, pad_w = pad.shape

		try:
			if (x_edge[0][0] < (w/2) and x_sum[x_edge[0][0]] < 50):
				l = x_edge[0][0] ; r = w ; cur_label=1
			if (x_edge[0][-1] > (w/2) and x_sum[x_edge[0][-1]] < 50):
				l = 0 ; r = x_edge[0][-1] ; cur_label=1
			elif(x_edge[0][0] > (w/2+20) and x_sum[x_edge[0][0]] < 50): 
				l = x_edge[0][0]; r = w; cur_lable=0
			elif(x_edge[0][-1] < (w/2-20) and x_sum[x_edge[0][-1]] < 50): 
				l = 0 ; r = x_edge[0][-1]; cur_label=0
			else:
				l = 0 ; r = w ; cur_label=1
		except:
			l = 0 ; r = w ; cur_label=1
		try:
			t = y_edge[0][0]+50; b = h
			self.prev_y = t
		except:
			t = self.prev_y; b = h
		
		#cv2.rectangle(image, (l,t), (r,b), (1,1,1) ,2)
		
		'''creates video of faces only'''
		try:
			if(l<w/2 and x_sum[x_edge[0][-1]] < 50):
				pad[0:org_img[h-pad_h:h, 0:r].shape[0], w-r:w] = org_img[h-pad_h:h, 0:r]
			elif(r>w/2 and x_sum[x_edge[0][0]] < 50):
				pad[0:org_img[h-pad_h:h, 0:r].shape[0], 0:w-l] = org_img[h-pad_h:h, l:r]
			else:
				pad[0:org_img[h-pad_h:h, 0:r].shape[0], l:r] = org_img[h-pad_h:h, l:r]
		except:
			pad[0:org_img[h-pad_h:h, 0:r].shape[0], l:r] = org_img[h-pad_h:h, l:r]
		self.prev_frame = pad

		return pad, cur_label

class non_linearity:

	def __init__(self, std_sw, std_sh, std_t):
		self.std_sw = std_sw
		self.std_sh = std_sh
		self.std_t = std_t

	def act(self, x): 
		return np.sign(x)*np.log(1+np.abs(x))

	def gaus_act(self, frames):
		frames_diff = np.diff(frames, axis=0)
		#print("np diff-sum-divide done..", frames_diff.shape)
		#frames = block_reduce(frames, block_size=(1, 2, 2), func=np.max)
		#print(frames_diff.shape, "pooling done..")
		#frames_diff = ndimage.gaussian_filter(frames_diff,sigma=(self.std_t,self.std_sh,self.std_sw))
		print("gaussina blur done..")
		#frames_diff = self.act(frames_diff)
		#frames = self.act(frames)
		#print("activation done")
		'''
		frames_diff = np.uint8((((frames_diff-frames_diff.min()))/frames_diff.max())*255)
		frames = np.uint8((((frames-frames.min()))/frames.max())*255)
		'''
		#frames_diff = (frames_diff-frames_diff.min())/frames_diff.max()
		#frames = (frames-frames.min())/frames.max()

		#print(frames.shape, frames_diff.shape)
		#creating a stack of images to have h,w,ch dimension
		#frames_out = []
		#for i in range(frames_diff.shape[0]):
		#	if(i == 0):
		#		frames_out.append(np.stack((frames[i], frames[i], frames_diff[i]), axis=0))
		#	else:
		#		frames_out.append(np.stack((frames_diff[i-1], frames[i], frames_diff[i]), axis=0))
		#frames_out = np.array(frames_out)

		return frames_diff


if __name__ == '__main__':
	
	path = '../dlib_model/shape_predictor_68_face_landmarks.dat'
	pf = preprocess_face()
	std_sw = 4 #aprox pixel distance between the eyes
	std_sh = 4 #aprox pixel distance between head and upper lip
	std_t = 3 #temporal sigma
	nl = non_linearity(std_sw, std_sh, std_t)

	#load mat file (sample for testing)
	#file_list = glob.glob('../data/mat_files/*')
	#file_list = glob.glob('../data/corrupt_frames/*')
	file_list = ['../data/mat_files/BOSS_-BOSS_146_1_3_cm-190_22_33_10_257_IR.mat']
	#file_list = ['../data/mat_files/BOSS_-BOSS_149_1_2_cr-291_16_30_27_010_IR.mat']
	#file_list = ['../data/mat_files/BOSS_-BOSS_126_1_3_cr-081_21_40_09_108_IR.mat']

	for v_file_path in file_list:
		org_frames = []
		label = []
		nm = v_file_path.split('/')[-1]
		fname2 = f'../vid_data/org_{nm}.avi'

		video_file = h5py.File(v_file_path,'r')
		print(video_file)
		img = video_file['data']
		print(img.shape)

		for idx in range(img.shape[0]-300):
			print(idx)
			f = pf.face_tube(idx, img)
			org_frames.append(f)
			#label.append(lb)
		#print(len(label), len(org_frames))
		org_frames = np.array(org_frames)
		#import pdb; pdb.set_trace()
		#org_frames = nl.gaus_act(org_frames)
		#frames = np.rollaxis(frames, 1, 3)
		#org_frames = np.rollaxis(frames, 3, 2)
		org_frames = np.uint8(org_frames*255)
		num, h, w = org_frames.shape
		print("shape", org_frames.shape)
		
		out2 = cv2.VideoWriter(fname2, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h), isColor=False)
		for i in range(org_frames.shape[0]):
			#print(frame.shape)
			out2.write(np.uint8(org_frames[i]))
		out2.release()

