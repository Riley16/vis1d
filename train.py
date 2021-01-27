import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import normal
Normal = normal.Normal
from torch.distributions import uniform
Uniform = uniform.Uniform
import numpy as np
import math
import os

import matplotlib.pyplot as plt

from models.predictiveAE import PredictiveAE
from env.env1D import Env1DObject, SineWave, AdditiveGaussian, Rigid1D
from options import parser

import pdb


'''
simple experiment for individuating "discrete objects" on 1D number line with background
really want to model flexible featural reference frames and individuation as discontinuity in a feature space
individuation is the picking out of some items or entities against a background or other entities

entities being individuated often are simple physical objects with object permenance, 


look into image segmentation and object detection models in deep learning, although object detection is a different task from individuation
explore model representations with pretrained models
also numerosity models like what I read in Jay's class, but with more modern architectures, etc.
look individuation models straightup (object detection in deep learning roughly maps onto individuation)

need to also think about object representational structure more generally and not focus solely on individuation,
try to think of additional metrics and experimental extensions for the current WWI model, ask what Randy is doing/planning


simple mathematical backgrounds like sinusoids, smaller objects, could have motion
main objects to be tracked could be larger discrete objects riding on top of the waves




could include occluding objects as larger bars as additional predictive challenge

questions of how to focus predictive learning on main objects to predict their motions while ignoring background to an extent
	human interest in objects against background is likely driven not only by predictive learning, but also by reward
	Leabra bidirectionality/RNN could immediately provide simple iconic memory to account for static background objects in part
	would still have learning focused on large background with simple ML losses like L2
	would need some kind of attention, task?
should include some trials with static backgrounds/objects


environment of 1D number line. can have simple kinematic objects and background textures
could implement in Saccade environment with saccading disabled
would need to create simple block object files, add overlay for textured backgrounds
just check into textured backgrounds, removing any Gabor filtering on input image
make environment shape 1D
convert to torch for backprop modeling

deep learning model:
want to preserve feedforward/feedback structure in predictive coding, but will need to eliminate feedback loops for the most part beyond RNNs


column-wise nets


TODO

env/model parameter config files storage
model checkpointing, model metrics
model run management
unit tests

make into repository for environment (maybe include models) once I've cleaned things up, separated out personal notes, added some more functionality,
aggregate model metrics https://www.tensorflow.org/tensorboard/dataframe_api
mlflow

'''

def train(args):
	full_exp_dir = os.path.join(args.log_dir, args.exp_dir)
	logger = SummaryWriter(full_exp_dir)

	step_sec = 0.5

	env_size = args.env_size

	objs = dict()
	# objs['background'] = SineWave(name='background',
	# 							  env_size=env_size,
	# 							  spatial_freq_dist=1.0,
	# 							  amp_dist=1.0,
	# 							  vel_dist=0.1)

	rel_size = 0.1
	# objs['rigid_quad1'] = Rigid1D(name='rigid_quad1',
	# 							  form=3 + torch.linspace(0, 2, int(rel_size*env_size))**2,
	# 							  size_dist=rel_size,
	# 							  env_size=env_size,
	# 							  pos_dist=0.0,
	# 							  vel_dist=0.1)

	objs['rigid_rect1'] = Rigid1D(name='rigid_rect1',
								  form=5 * torch.ones(int(rel_size*env_size)),
								  size_dist=rel_size,
								  env_size=env_size,
								  pos_dist=Normal(0.5, 0.25),
								  vel_dist=Normal(0.0, 0.1))

	pole_rel_size = 0.15
	# objs['fixed_pole1'] = Rigid1D(name='fixed_pole1',
	# 							  form=7 * torch.ones(int(pole_rel_size*env_size)),
	# 							  size_dist=pole_rel_size,
	# 							  env_size=env_size,
	# 							  pos_dist=Normal(loc=0.5, scale=0.25),  #0.5,
	# 							  vel_dist=0.0)

	objs['noise'] = AdditiveGaussian(name='noise',
									 size_dist=1.0,
									 env_size=env_size,
									 std_dist=0.1)

	env = Env1DObject(size=env_size, objs=objs)
	model = PredictiveAE(size=env_size, num_lstm_layers=1)
	loss_func = nn.MSELoss()
	opt = torch.optim.Adam(params=model.parameters(), lr=0.003)

	trial_step_log_num = 0

	for trial in range(args.trials):
		trial_metrics = {'loss_step/train': []}
		env.reset()
		model.reset()
		state = env.get_state()
		for step in range(args.steps_per_trial):
			pred = model(state)
			env.step()
			next_state = env.get_state()
			loss = loss_func(pred, next_state)
			opt.zero_grad()
			loss.backward(retain_graph=True)
			opt.step()

			# rendering
			if args.render and trial % args.video_freq == 0 and trial >= 0:
				plt.clf()
				plt.plot(np.linspace(0, 1, env_size), state[0], label='True curr')
				plt.plot(np.linspace(0, 1, env_size), next_state[0], label='True next')
				plt.plot(np.linspace(0, 1, env_size), pred[0].detach().numpy(), label='Pred next')
				plt.legend()
				if step == 0:
					plt.show(block=False)
				plt.pause(step_sec)
				if step == args.steps_per_trial - 1:
					plt.close()

			state = next_state

			# step-based logging
			if trial % args.print_freq == 0:
				trial_step_log_num += 1
				logger.add_scalar('loss_step/train', loss, trial_step_log_num)
				trial_metrics['loss_step/train'].append(loss)

		# trial-based logging
		if trial % args.print_freq == 0:
			trial_metrics['loss_trial/train'] = torch.mean(torch.Tensor(trial_metrics['loss_step/train']))
			print('trial: {}\ttrain loss: {:.5}'.format(trial, trial_metrics['loss_trial/train']))
			logger.add_scalar('loss_trial/train', trial_metrics['loss_trial/train'], trial)

		if trial % args.save_freq == 0 and trial > 0:
			pass


if __name__ == '__main__':
	args = parser.parse_args()
	train(args)
