import numpy as np 
import torch
import torch.nn as nn

softmax=nn.Softmax(-1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class loss_pep():
	def __init__(self, num_bins=33, max_val=1):
		self.num_bins = num_bins
		self.max_val  = max_val
		self.bins	  = np.linspace(0, 1,num_bins)
		self.class_criterion = nn.CrossEntropyLoss()
		self.reg_criterion   = nn.MSELoss()

	def __call__(self, pred, labels):
		#gt_labels = np.concatenate(labels) #fix it later
		labels    = labels.squeeze()
		gt_bins   = np.digitize(labels, self.bins, right=True)
		
		loss_class = self.class_loss(pred, torch.tensor(gt_bins).to(device))
		loss_reg   = self.reg_loss(pred, torch.tensor(labels).to(device))

		loss_total = 0.6*loss_class + 0.4*loss_reg

		return loss_total

	def class_loss(self, pred, gt):
		return self.class_criterion(pred, gt)

	def reg_loss(self, pred, gt):
		bin_numbers     = torch.tensor(self.bins).to(device).unsqueeze(0)
		expected_values = torch.sum(softmax(pred)*bin_numbers,1)

		return self.reg_criterion(expected_values, gt)

def predict_pep(pred, num_bins=33):
	bins = np.linspace(0, 1,num_bins)
	bin_numbers = torch.tensor(bins).to(device).unsqueeze(0)
	expected_values = torch.sum(softmax(pred)*bin_numbers, 1)

	return expected_values
	
