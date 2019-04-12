import torch
import numpy
import gym
from collections import deque
from model import Actor , Critic
import random
import torch.optim as optim
env = gym.make('Pendulum-v0')

LOSS = torch.nn.MSELoss()
TAU = 0.001
A_LR = 0.0001
C_LR = 0.001
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 64
MAX_EPISODES = 50000
MAX_EPISODES_STEPS = 1000
GAMMA = 0.99
BUFFER = deque(maxlen = BUFFER_SIZE)
actor  = Actor()
critic = Critic()
Critic_T = Critic()
Actor_T = Actor()

Actor_optim = optim.Adam(actor.parameters(),lr = A_LR)
Critic_optim  = optim.Adam(critic.parameters(),lr = C_LR)

def train():

	sample = random.sample(BUFFER,MINIBATCH_SIZE)
	Y = []
	states =[]
	actions = []
	for cnt,row in enumerate(sample):
		state = row[0]
		action = row[1]
		reward = row[2]
		next_state = row[3]
		is_done = row[4]
		states.append(state)
		actions.append([action])
		S_action = Actor_T.forward(next_state).item()
		Q_S = Critic_T.forward(next_state,numpy.array([S_action])).item()
		if not is_done:
			Y.append(reward+ GAMMA*Q_S)
		else:
			Y.append(reward)

	
	Q_s = critic.forward(numpy.array(states),numpy.array(actions))
	loss_critic = LOSS(torch.Tensor(Y),Q_s)
	S = numpy.array(states)
	loss_actor =  -critic.forward(S,actor.forward(S))
	loss_actor.backward()
	loss_critic.backward()
	Actor_optim.step()
	Critic_optim.step()



episodes_cnt = 0
while(episodes_cnt <= MAX_EPISODES): 	 
	episodes_cnt +=1
	frame = env.reset()
	is_done = False
	step_cnt = 0
	while (not is_done and step_cnt < MAX_EPISODES_STEPS):
		step_cnt += 1
		actor.eval()
		action = actor.forward(frame)[0].item()
		new_frame,reward,is_done,_ = env.step([action])
		BUFFER.append((frame,action,reward,new_frame,is_done))
		if len(BUFFER) > 64:
			train()
