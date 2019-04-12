import  torch
import torch.nn as nn
import numpy
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self):
		super(Actor,self).__init__()
		self.fc1 = nn.Linear(3,400)
		self.fc2 = nn.Linear(400,300)
		# self.batch_norm1 = nn.BatchNorm1d(400)
		# self.batch_norm2 = nn.BatchNorm1d(300)
		self.fc3 = nn.Linear(300,1)

	def forward(self, X):
		dim = X.ndim
		X = torch.from_numpy(X)
		if dim == 1:
			X = X.unsqueeze(0)

		X = X.type(torch.FloatTensor)
		X = self.fc1(X)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)
		X = self.fc3(X)
		X = F.tanh(X)*2
		return X

class Critic(nn.Module):

	def __init__(self):
		super(Critic,self).__init__()
		self.fc_obs_1 = nn.Linear(3,400)
		self.fc_obs_2 = nn.Linear(400,300)
		# self.batch_norm1 = nn.BatchNorm1d(400)
		self.fc_action_1 = nn.Linear(1,300)
		self.final = nn.Linear(300,1)

	def forward(self, OBS, ACTION):
		dim = OBS.ndim
		if dim == 1:
			OBS = torch.from_numpy(OBS).type(torch.FloatTensor).unsqueeze(0)
			ACTION = torch.from_numpy(ACTION).type(torch.FloatTensor).unsqueeze(0)
		else:
			OBS = torch.from_numpy(OBS).type(torch.FloatTensor)
			ACTION = torch.from_numpy(ACTION).type(torch.FloatTensor)

		OBS = self.fc_obs_1(OBS)
		OBS = F.relu(OBS)
		OBS = self.fc_obs_2(OBS)
		ACTION = self.fc_action_1(ACTION)
		Q = F.relu(torch.add(OBS,ACTION))
		Q = self.final(Q)

		return Q


