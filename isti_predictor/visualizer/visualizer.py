import numpy as np
import cv2
import torch
import os
import sys
import random
import torch.nn as nn
import torch.utils.data as tdata
import glob
from matplotlib import pyplot as plt

sys.path.append(".")
from visualizer_dataloader import thermaldataset


def visualize(data_sample):
	data = data_sample['data']
	label = data_sample['label']
	feetin_frame = data_sample['feetin_frame']
	feetin_ts = data_sample['feetin_ts']
	vid_fname = data_sample['filename']
	print(vid_fname[0])
	fname = f'../vid_data/{vid_fname[0]}.avi'
	_, d, h, w = data.shape
	out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h), isColor=True)
	print(data.numpy().shape, label.shape, feetin_frame, feetin_ts)
	f_count = 0
	for i in range(d):
		vid_i = data[0][i].numpy()
		ecg_i = label[0][i].numpy().flatten()
		fig, ax = plt.subplots(figsize=(2.4, 1.6))
		ax.plot(ecg_i)
		fig.canvas.draw()
		np_plot = np.array(fig.canvas.renderer.buffer_rgba())
		vid_i   = cv2.cvtColor(vid_i, cv2.COLOR_GRAY2BGR)
		#np_plot = cv2.cvtColor(np_plot, cv2.CV_BGRA2HSV)
		#print("shape of plot and img", np_plot.shape, vid_i.shape)
		vid_i[0:160,:,:] = np_plot[:,:,0:3]
		if(i == feetin_frame-4): f_count = 15
		if(f_count>0):
			cv2.putText(vid_i, 'FeetIn Water', (160,120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255) ,\
						2, cv2.LINE_AA)
			f_count = f_count-1
		plt.close()
		out.write(vid_i)
	out.release()
	return

	
#file usage : python visualizer.py ../data/test_label ../data/mat_files ../data/sync_data 

if __name__=='__main__':
	label_name    = sys.argv[1]
	ir_vid_name   = sys.argv[2]
	sync_sig_name = sys.argv[3]
	
	print(label_name, ir_vid_name)
	label    = "{}/".format(label_name)
	ir_video = "{}/".format(ir_vid_name)
	print(label, ir_video)
	
	
	visualize_dataset = thermaldataset(
        label    = "{}/".format(label_name), 
        ir_video = "{}/".format(ir_vid_name), 
        sync_sig = "{}/".format(sync_sig_name), 
        phase='train'
    )

	trainloader = torch.utils.data.DataLoader(visualize_dataset,batch_size=1,shuffle=True,num_workers=1)
	for data_sample in trainloader:
		try:
			if(data_sample == -1):
				print("Value -1 returned")
				continue
		except:
			pass
		visualize(data_sample)
