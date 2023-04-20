import numpy as np 
import torch
import torch.nn as nn

softmax=nn.Softmax(-1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class loss_pep():
	def __init__(self, num_bins=1, max_val=1):
		self.num_bins = num_bins
		self.max_val  = max_val
		self.bins	  = np.linspace(0, 1,num_bins)
		self.class_criterion = nn.CrossEntropyLoss()
		self.reg_criterion	 = nn.MSELoss()
		self.BCE	  = nn.BCELoss()

	def __call__(self, pred, pred2, labels, s_label):
		#gt_labels = np.concatenate(labels) #fix it later
		labels	  = labels.squeeze().to(device).to(torch.float32)
		s_label	  = s_label.to(device).to(torch.float32)
		#gt_bins   = np.digitize(labels, self.bins, right=True)
		
		#loss_class = self.class_loss(pred, torch.tensor(gt_bins).to(device))
		loss_reg    = self.reg_loss(pred, labels)
		loss_shrink = self.shrinkage_loss(pred, labels)
		loss_stress = self.BCE(pred2, s_label)

		#loss_total = loss_class + 0.2*loss_reg
		#stress_loss = 
		loss_total = loss_reg + loss_shrink + 0.4*loss_stress

		return loss_total

	def class_loss(self, pred, gt):
		return self.class_criterion(pred, gt)

	def reg_loss(self, pred, gt):
		#bin_numbers		= torch.tensor(self.bins).to(device).unsqueeze(0)
		#expected_values = torch.sum(softmax(pred),1)
		expected_values = pred.squeeze()

		return self.reg_criterion(expected_values, gt)

	def shrinkage_loss(self, pred, gt):
		pred = pred.squeeze()
		abs_error = (pred - gt).abs()
		loss = (torch.exp(gt) * (abs_error.pow(2)))/ (1 + torch.exp(10 * (0.2-abs_error)))
		loss = loss.mean()
		
		return loss

def predict_pep(pred, num_bins=1):
	#bins = np.linspace(0, 1,num_bins)
	#bin_numbers = torch.tensor(bins).to(device).unsqueeze(0)
	#expected_values = torch.sum(softmax(pred)*bin_numbers, 1)

	return pred
	
