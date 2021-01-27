import torch
from torch import nn
from torch.distributions import normal
Normal = normal.Normal
from torch.distributions import uniform
Uniform = uniform.Uniform
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt


class Env1DObject:
	def __init__(self, objs=dict(), size=128):
		self.objs = objs
		self.size = size

	def add_obj(self, obj):
		assert obj.name not in list(self.objs.keys()), 'Object name "{}" is already in use'.format(obj.name)
		self.objs[obj.name] = obj

	def get_state(self):
		self.state = torch.zeros(self.size)
		with torch.no_grad():
			for obj in self.objs.values():
				# for now, objects simply occlude each other
				pos = int(obj.pos * self.size)
				state = obj.get_state()

				env_pos0 = pos
				env_pos1 = pos + obj.element_size
				if pos < 0:
					env_pos0 = 0
					state = state[-pos:]
				if pos + obj.element_size <= 0:
					env_pos1 = 0
				elif pos + obj.element_size > self.size:
					state = state[:-(pos + obj.element_size - (self.size))]
					env_pos1 = self.size

				if obj.mode == 'max':  # for treating 1D array element values as depths
					self.state[env_pos0:env_pos1] = torch.max(state, self.state[env_pos0:env_pos1])
				elif obj.mode == 'add':  
					self.state[env_pos0:env_pos1] += state
				elif obj.mode == 'equal':  # for treating 1D array element values as arbitrary, abstract feature values, e.g. a color or texture dimension
					self.state[env_pos0:env_pos1] = state
				else:
					assert False, 'Only Object1D modes "max", "add", and "equal" are supported'

		return self.state.reshape(1, self.state.shape[0])

	def step(self):
		with torch.no_grad():
			for obj in self.objs.values():
				obj.step()

	def reset(self):
		with torch.no_grad():
			for obj in self.objs.values():
				obj.reset()


class Object1D:
	def __init__(self, name, pos_dist=0.0, mode='max'):
		self.name = name
		# relative position in environment of leftmost element 
		# with 0.0 being the leftmost environment position and 1.0 being the rightmost 
		self.pos_dist = pos_dist
		self.mode = mode

	def get_state(self):
		raise NotImplemented

	def step(self):
		raise NotImplemented

	def reset(self):
		for name, attr in list(self.__dict__.items()):
			if name[-5:] == '_dist':
				if isinstance(attr, torch.distributions.Distribution):
					self.__dict__[name[:-5]] = attr.sample(torch.Size([1]))
				else:
					self.__dict__[name[:-5]] = attr


class SineWave(Object1D):
	def __init__(self, name, 
				 size_dist=1.0, 
				 env_size=128, 
				 spatial_freq_dist=1.0,
				 t_freq_dist=1.0, 
				 amp_dist=1.0, 
				 standing=False, 
				 vel_dist=0.1, 
				 phase_dist=0.0, 
				 pos_dist=0.0,
				 mode='max'):
		'''
		all parameters having the suffix "_dist" can be input as either a torch.distribution.Distribution object for random sampling
		or as a single parameter value which will be used for all instantiations of parameter values upon reset
		'''

		super(SineWave, self).__init__(name, pos_dist=pos_dist, mode=mode)
		self.size_dist = size_dist
		self.env_size = env_size
		# in units of cycles per self.size elements
		self.spatial_freq_dist = spatial_freq_dist
		# only used for standing wave
		self.t_freq_dist = t_freq_dist
		self.amp_dist = amp_dist
		self.standing = standing
		# in self.size units
		self.vel_dist = vel_dist
		# radians
		self.phase_dist = phase_dist
		self.reset()

	def get_state(self):
		if self.standing:
			raise NotImplemented
		else:
			self.state = torch.zeros(self.element_size)
			self.state = self.amp + self.amp * \
					torch.sin(2*math.pi*torch.linspace(0, self.spatial_freq, self.element_size) + self.phase)
		return self.state

	def step(self):
		self.phase -= 2*math.pi*self.spatial_freq*self.vel

	def reset(self):
		super(SineWave, self).reset()
		self.element_size = int(self.size * self.env_size)


class AdditiveGaussian(Object1D):
	def __init__(self, name, size_dist=1.0, env_size=128, std_dist=1.0, pos_dist=0.0):
		super(AdditiveGaussian, self).__init__(name, pos_dist, 'add')
		self.size_dist = size_dist
		self.env_size = env_size
		self.std_dist = std_dist
		self.reset()

	def get_state(self):
		return self.state

	def step(self):
		self.state = torch.distributions.normal.Normal(0, self.std).sample(torch.Size([self.element_size]))

	def reset(self):
		super(AdditiveGaussian, self).reset()
		self.element_size = int(self.size * self.env_size)
		self.state = torch.distributions.normal.Normal(0, self.std).sample([self.element_size])


class Rigid1D(Object1D):
	def __init__(self, name, form, size_dist=0.2, env_size=128, pos_dist=0.0, vel_dist=0.1, min_visible_steps=5, mode='max'):
		super(Rigid1D, self).__init__(name, pos_dist, mode)
		self.size_dist = size_dist
		self.env_size = env_size
		assert not isinstance(size_dist, torch.distributions.Distribution), 'Rigid1D object size must be fixed to size of form and cannot be stochastic'
		self.element_size = int(self.size_dist * self.env_size)
		assert form.shape[0] == self.element_size, 'form size {} and element_size {} are not equal.'.format(form.shape[0], self.element_size)
		# center of object
		self.vel_dist = vel_dist
		# shape of 1D object
		self.state = form
		self.min_visible_steps = min_visible_steps
		self.reset()

	def get_state(self):
		return self.state

	def step(self):
		self.pos += self.vel

	def reset(self):
		super(Rigid1D, self).reset()

		# rejection sampling to ensure object stays in field of vision for at least self.min_visible_steps steps
		while self.pos + self.vel * self.min_visible_steps > 1.0 or self.pos + self.vel * self.min_visible_steps < 0.0:
			if isinstance(self.vel_dist, torch.distributions.Distribution):
				self.vel = self.vel_dist.sample(torch.Size([1]))
			else:
				self.vel = self.vel_dist
			if isinstance(self.pos_dist, torch.distributions.Distribution):
				self.pos = self.pos_dist.sample(torch.Size([1]))
			else:
				self.pos = self.pos_dist

'''
TODO
object wrappers (for individualized noise distributions, feature overlays, transforms, etc.)
config files
optimize rendering
unit tests
add data precomputation option
prevent objects from leaving environment with something more principled than 
	rejection sampling to ensure objects stay in env for some fixed min number of steps?
saccade motion/planning
heteroskedasticity for additive noise background
static additive noise to model complex local feature variations which are fixed over time


'''

if __name__ == '__main__':
	trials = 2
	env_size = 128
	steps_per_trial = 8
	manual_render_step = False

	# visualization of environment as video
	render = True
	step_sec = 1.0

	objs = dict()
	objs['background'] = SineWave('background', env_size=env_size, spatial_freq_dist=1.0, amp_dist=1.0, vel_dist=0.05)

	rel_size = 0.12
	# objs['rigid_quad1'] = Rigid1D(name='rigid1', form=3 + torch.linspace(0, 2, int(rel_size*env_size))**2, size_dist=rel_size, env_size=env_size, 
	# 							  pos_dist=0.0,  #Normal(0.5, 0.25),
	# 							  vel_dist=0.1)  #Normal(0.0, 0.25))

	objs['rigid_rect1'] = Rigid1D(name='rigid1', form=6 * torch.ones(int(rel_size*env_size)), size_dist=rel_size, env_size=env_size, pos_dist=0.0, vel_dist=0.1)

	pole_rel_size = 0.2
	objs['fixed_pole'] = Rigid1D(name='rigid1',
								 form=10 * torch.ones(int(pole_rel_size*env_size)),
								 size_dist=pole_rel_size,
								 env_size=env_size,
								 pos_dist=0.5,  #torch.distributions.normal.Normal(loc=0.3, scale=0.15),
								 vel_dist=0.0)

	objs['noise'] = AdditiveGaussian(name='noise', size_dist=1.0, env_size=env_size, std_dist=0.1)

	env = Env1DObject(size=env_size, objs=objs)

	for trial in range(trials):
		env.reset()
		for step in range(steps_per_trial):
			state = env.get_state()
			if render:
				plt.clf()
				plt.plot(np.linspace(0, 1, env_size), state[0])
				if step == 0:
					plt.show(block=False)
				elif manual_render_step:
					plt.show()
				plt.pause(step_sec)
				if step == steps_per_trial - 1:
					plt.close()
			env.step()
