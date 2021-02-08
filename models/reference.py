import torch
import torch.nn as nn

class NeuralNet(nn.Module):
	def __init__(self, in_size, out_size, dropout):

		super(NeuralNet, self).__init__()

		self.layers = nn.Sequential(
		nn.Linear(in_size, 256),
		nn.ReLU(),
		nn.Dropout(p=dropout),
		nn.Linear(256, 128),
		nn.ReLU(),
		nn.Dropout(p=dropout),
		nn.Linear(128, 64),
		nn.ReLU(),
		nn.Dropout(p=dropout),
		nn.Linear(64, 32),
		nn.ReLU(),
		nn.Dropout(p=dropout),
		nn.Linear(32, out_size))

	def forward(self, x):
		
		for layer in self.layers:
			x = layer(x)

		return x