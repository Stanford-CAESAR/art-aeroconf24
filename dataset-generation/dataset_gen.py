import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt
import random as rnd

from dynamics.orbit_dynamics import *
from optimization.rpod_scenario import *
from optimization.ocp import *

N_data = 200000

n_S = 6 # state size
n_A = 3 # action size

states_roe_cvx = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m]
states_rtn_cvx = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
actions_cvx = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float) # [m/s]

states_roe_scp = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m]
states_rtn_scp = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float) # [m,m,m,m/s,m/s,m/s]
actions_scp = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float) # [m/s]

horizons = np.empty(shape=(N_data, ), dtype=float)
dtime = np.empty(shape=(N_data, ), dtype=float)
time = np.empty(shape=(N_data, n_time_rpod), dtype=float)
oe = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float)

i_unfeas = []

for i in range(N_data):
    
    horizon = np.linspace(1,3, 100) # [# orbits] - transfer horizon
    da = np.linspace(-5, 5, 100) # [m]
    dlambda = np.linspace(-100, 100, 100) # [m]
    de = np.linspace((1/E_koz.item((0,0)))+5, (1/E_koz.item((0,0)))+30, 100)
    di = np.linspace((1/E_koz.item((2,2)))+5, (1/E_koz.item((2,2)))+30, 100)
    ph_de = np.pi/2 + np.linspace(-5, 5, 100)*np.pi/180 # [m]
    ph_di = np.pi/2 + np.linspace(-5, 5, 100)*np.pi/180 # [m]
        
    # Pick horizon
    hrz_i = horizon[rnd.randint(0, 100-1)]

    #  Pick initial passively safe relative orbit around the space station
    da_i = da[rnd.randint(0, 100-1)]
    dlambda_i = dlambda[rnd.randint(0, 100-1)]
    de_i = de[rnd.randint(0, 100-1)]
    di_i = di[rnd.randint(0, 100-1)]
    ph_de_i = ph_de[rnd.randint(0, 100-1)]
    ph_di_i = ph_di[rnd.randint(0, 100-1)]

    state_roe_0 = np.array([da_i, dlambda_i, de_i*np.cos(ph_de_i), de_i*np.sin(ph_de_i), di_i*np.cos(ph_di_i), di_i*np.sin(ph_di_i)]).reshape((6,))

    # Dynamics Matrices Precomputations
    stm_i, cim_i, psi_i, oe_i, time_i, dt_i = dynamics_roe_optimization(oe_0_ref, t_0, hrz_i, n_time_rpod)

    # Solve transfer cvx
    states_roe_cvx_i, actions_cvx_i, feas_cvx_i = ocp_cvx(stm_i, cim_i, psi_i, state_roe_0, n_time_rpod)
    states_rtn_cvx_i = roe_to_rtn_horizon(states_roe_cvx_i, oe_i, n_time_rpod)

    if np.char.equal(feas_cvx_i,'optimal'):

        #  Solve transfer scp
        states_roe_scp_i, actions_scp_i, feas_scp_i = solve_scp(stm_i, cim_i, psi_i, state_roe_0, states_roe_cvx_i, n_time_rpod)
        states_rtn_scp_i = roe_to_rtn_horizon(states_roe_scp_i, oe_i, n_time_rpod)

        if np.char.equal(feas_scp_i,'optimal'):
            states_roe_cvx[i,:,:] = np.transpose(states_roe_cvx_i)
            states_rtn_cvx[i,:,:] = np.transpose(states_rtn_cvx_i)
            actions_cvx[i,:,:] = np.transpose(actions_cvx_i)
             
            states_roe_scp[i,:,:] = np.transpose(states_roe_scp_i)
            states_rtn_scp[i,:,:] = np.transpose(states_rtn_scp_i)
            actions_scp[i,:,:] = np.transpose(actions_scp_i)
        
            horizons[i] = hrz_i
            dtime[i] = dt_i
            time[i,:] = np.transpose(time_i)
            oe[i,:,:] = np.transpose(oe_i)
        else:
            i_unfeas += [ i ]
    else:
        i_unfeas += [ i ]

# Remove unfeasible data points
if i_unfeas:
    states_roe_cvx = np.delete(states_roe_cvx, i_unfeas, axis=0)
    states_rtn_cvx = np.delete(states_rtn_cvx, i_unfeas, axis=0)
    actions_cvx = np.delete(actions_cvx, i_unfeas, axis=0)
    
    states_roe_scp = np.delete(states_roe_scp, i_unfeas, axis=0)
    states_rtn_scp = np.delete(states_rtn_scp, i_unfeas, axis=0)
    actions_scp = np.delete(actions_scp, i_unfeas, axis=0)
    
    horizons = np.delete(horizons, i_unfeas, axis=0)
    dtime = np.delete(dtime, i_unfeas, axis=0)
    time = np.delete(time, i_unfeas, axis=0)
    oe = np.delete(oe, i_unfeas, axis=0)

#  Save dataset
np.savez_compressed('dataset/dataset-rpod-v05-scp', states_roe_scp = states_roe_scp, states_rtn_scp = states_rtn_scp, actions_scp=actions_scp)
np.savez_compressed('dataset/dataset-rpod-v05-cvx', states_roe_cvx = states_roe_cvx, states_rtn_cvx = states_rtn_cvx, actions_cvx=actions_cvx)
np.savez_compressed('dataset/dataset-rpod-v05-param', time = time, oe = oe, dtime = dtime, horizons = horizons)