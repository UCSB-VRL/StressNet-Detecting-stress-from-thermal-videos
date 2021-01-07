import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class pep_detector(nn.Module):

	def __init__(self, in_channel, scale, num_bins=33):
		super(pep_detector, self).__init__()
		self.representation_size = in_channel*scale
		
		self.l1 = nn.Linear(in_channel, self.representation_size)
		self.l2 = nn.Linear(self.representation_size, num_bins)

	def forward(self, x):
		x = F.relu(self.l1(x))
		x = self.l2(x)

		return x

if __name__=='__main__':
	model = pep_detector(2, 4).to(device)
	data  = torch.rand(1,240,2).float().to(device)
	import pdb; pdb.set_trace()
	data = data.squeeze()
	out   = model(data)


