import os
import sys
import argparse

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

from dynamics.orbit_dynamics import *
from optimization.rpod_scenario import *
from optimization.ocp import *

parser = argparse.ArgumentParser(description='transformer-rpod')
parser.add_argument('--data_dir', type=str, default='dataset',
                    help='defines directory from where to load files')
args = parser.parse_args()
args.data_dir = root_folder + '/' + args.data_dir

print('Loading data...', end='')

data_scp = np.load(args.data_dir + '/dataset-rpod-v05-scp.npz')
data_cvx = np.load(args.data_dir + '/dataset-rpod-v05-cvx.npz')
data_param = np.load(args.data_dir + '/dataset-rpod-v05-param.npz') 

time_discr = data_param['dtime']
oe = data_param['oe']
n_data = oe.shape[0]
n_time = oe.shape[1]  # this excludes the target state time

# Pre-compute torch states roe and rtn
states_roe_scp = data_scp['states_roe_scp']
torch_states_roe_scp = torch.from_numpy(states_roe_scp)
torch.save(torch_states_roe_scp, args.data_dir + '/torch_states_roe_scp.pth')

states_rtn_scp = data_scp['states_rtn_scp']
torch_states_rtn_scp = torch.from_numpy(states_rtn_scp)
torch.save(torch_states_rtn_scp, args.data_dir + '/torch_states_rtn_scp.pth')

states_roe_cvx = data_cvx['states_roe_cvx']
torch_states_roe_cvx = torch.from_numpy(states_roe_cvx)
torch.save(torch_states_roe_cvx, args.data_dir + '/torch_states_roe_cvx.pth')

states_rtn_cvx = data_cvx['states_rtn_cvx']
torch_states_rtn_cvx = torch.from_numpy(states_rtn_cvx)
torch.save(torch_states_rtn_cvx, args.data_dir + '/torch_states_rtn_cvx.pth')

# Pre-compute torch actions
actions_scp = data_scp['actions_scp']
torch_actions_scp = torch.from_numpy(actions_scp)
torch.save(torch_actions_scp, args.data_dir + '/torch_actions_scp.pth')

actions_cvx = data_cvx['actions_cvx']
torch_actions_cvx = torch.from_numpy(actions_cvx)
torch.save(torch_actions_cvx, args.data_dir + '/torch_actions_cvx.pth')

# Pre-compute torch rewards to go and constraints to go
torch_rtgs_scp = torch.from_numpy(compute_reward_to_go(actions_scp, n_data, n_time))
torch.save(torch_rtgs_scp, args.data_dir + '/torch_rtgs_scp.pth')

torch_rtgs_cvx = torch.from_numpy(compute_reward_to_go(actions_cvx, n_data, n_time))
torch.save(torch_rtgs_cvx, args.data_dir + '/torch_rtgs_cvx.pth')

torch_ctgs_scp = torch.from_numpy(compute_constraint_to_go(states_rtn_scp, n_data, n_time))
torch.save(torch_ctgs_scp, args.data_dir + '/torch_ctgs_scp.pth')

torch_ctgs_cvx = torch.from_numpy(compute_constraint_to_go(states_rtn_cvx, n_data, n_time))
torch.save(torch_ctgs_cvx, args.data_dir + '/torch_ctgs_cvx.pth')