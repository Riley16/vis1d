import torch
from torch import nn
import pdb


class PredictiveAE(nn.Module):
	def __init__(self, size=128, num_lstm_layers=1, lstm_size=512):
		super(PredictiveAE, self).__init__()
		self.size = size
		self.lstm_size = lstm_size
		self.num_lstm_layers = num_lstm_layers

		self.lstms = [nn.LSTMCell(size, self.lstm_size)]
		self.c0s = [torch.zeros(1, self.lstm_size)]  #nn.Parameter(torch.zeros(1, self.lstm_size), requires_grad=True)
		self.h0s = [torch.zeros(1, self.lstm_size)]  #nn.Parameter(torch.zeros(1, self.lstm_size), requires_grad=True)
		self.hs = [[self.c0s[0]]]
		self.cs = [[self.h0s[0]]]

		for i in range(num_lstm_layers - 1):
			self.lstms.append(nn.LSTMCell(self.lstm_size, self.lstm_size))
			self.c0s.append(torch.zeros(1, self.lstm_size))  #nn.Parameter(torch.zeros(1, self.lstm_size), requires_grad=True)
			self.h0s.append(torch.zeros(1, self.lstm_size))  #nn.Parameter(torch.zeros(1, self.lstm_size), requires_grad=True)
			self.hs.append([self.c0s[i]])
			self.cs.append([self.h0s[i]])

		self.lin1 = nn.Linear(self.lstm_size, 256)
		self.lin2 = nn.Linear(256, size)
		self.relu = nn.ReLU()

	def forward(self, x, bp_time=False):
		self.hs[0], self.cs[0] = self.lstms[0](x, (self.hs[0], self.cs[0]))
		for i in range(1, self.num_lstm_layers):
			self.hs[i], self.cs[i] = self.lstms[i](self.hs[i-1], (self.hs[i], self.cs[i]))
		out = self.lin1(self.hs[self.num_lstm_layers-1])
		out = self.relu(out)
		out = self.lin2(out)
		if not bp_time:
			# TODO confirm backprop through time is being truncated on every time step for alpha-cycle-like predictive learning
			self.detach_state()
		return out

	def reset(self):
		with torch.no_grad():
			self.hs = self.h0s
			self.cs = self.c0s

	def detach_state(self):
		for i in range(self.num_lstm_layers):
			self.hs[i].detach()
			self.cs[i].detach()
