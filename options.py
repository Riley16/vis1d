import argparse

parser = argparse.ArgumentParser()

# logging
parser.add_argument('--log_dir', type=str, default='./logs', help='directory for logging within which directories (specified by "--exp_dir") '
																  'for individual experiments will be made')
parser.add_argument('--exp_dir', type=str, default='test', help='logging directory for individual experiment to be made in log_dir')
parser.add_argument('--print_freq', type=int, default=30, help='number of trials between logging/printing metrics to terminal')
parser.add_argument('--save_freq', type=int, default=500, help='number of trials between saving model checkpoints')
parser.add_argument('--render', action='store_true', help='Whether to periodically render model behavior')
parser.add_argument('--video_freq', type=int, default=100, help='number of trials between displaying pyplot plot videos of model behavior')

# training
parser.add_argument('--trials', type=int, default=500, help='number of trials per run')
parser.add_argument('--runs', type=int, default=1, help='number of runs in the experiment')
parser.add_argument('--steps_per_trial', type=int, default=8, help='number of recurrent update steps per trial')
parser.add_argument('--env_size', type=int, default=128, help='size in elements (array length) of 1D environment')
