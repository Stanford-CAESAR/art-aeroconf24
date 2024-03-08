import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(root_folder)
#print(sys.path)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt

from dynamics.orbit_dynamics import *
from rpod_scenario import *
from ocp import *
import transformer.manage as DT_manager

warmstart = 'both' # 'cvx'/'transformer'/'both'
scenario_test_dataset = True
state_representation = 'rtn' # 'roe'/'rtn'
dataset_to_use = 'scp' # 'scp'/'cvx'/'both'
transformer_ws = 'dyn' # 'dyn'/'ol'
transformer_model_name = 'checkpoint_rtn_ctgrtg' #'checkpoint_rtn_v11'/'checkpoint_rtn_v01'
select_idx = True # set to True to manually select a test trajectory via its index (idx)

results_dict = {

    'states_roe_ws_DT' : {},
    'states_rtn_ws_DT' : {},
    'actions_rtn_ws_DT' : {},
    'constr_DT' : {},
    'constr_viol_DT' : {},
    'cost_DT' : {},
    'ctg_true' : {},
}

datasets, dataloaders = DT_manager.get_train_val_test_data(state_representation, dataset_to_use, transformer_model_name)
test_dataset = datasets[2]
n_test = test_dataset.get_data_size()
for idx in range(n_test): 
    if not scenario_test_dataset:
        # Transfer horizon (orbits)
        hrz = 2
        # Initial relative orbit
        da = 0 # [m]
        dlambda = 75 # [m]
        de = 1/E_koz.item((0,0))+10
        di = 1/E_koz.item((2,2))+10
        ph_de = np.pi/2 + 0*np.pi/180; # [m]
        ph_di = np.pi/2 + 0*np.pi/180; # [m]
        state_roe_0 = np.array([da, dlambda, de*np.cos(ph_de), de*np.sin(ph_de), di*np.cos(ph_di), di*np.sin(ph_di)]).reshape((6,))
        relativeorbit_0 = roe_to_relativeorbit(state_roe_0, oe_0_ref)
    else:
        # Get the datasets and loaders from the torch data
        datasets, dataloaders = DT_manager.get_train_val_test_data(state_representation, dataset_to_use, transformer_model_name)
        train_loader, eval_loader, test_loader = dataloaders

        # Sample from test dataset
        if select_idx:
            test_sample = test_loader.dataset.getix(idx)
        else:
            test_sample = next(iter(test_loader))
        states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = test_sample
        print('Sampled trajectory ' + str(ix) + ' from test_dataset.')
        data_stats = test_loader.dataset.data_stats

        hrz = horizons.item()
        actions_i_unnorm = (actions_i * data_stats['actions_std']) + data_stats['actions_mean']
        if state_representation == 'roe':
            state_roe_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        elif state_representation == 'rtn':
            state_rtn_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
            state_roe_0 = map_rtn_to_roe(state_rtn_0, np.array(oe[0, :, 0]))
        #relativeorbit_0 = roe_to_relativeorbit(state_roe_0, oe_0_ref)

    # Dynamics Matrices Precomputations
    stm_hrz, cim_hrz, psi_hrz, oe_hrz, time_hrz, dt_hrz = dynamics_roe_optimization(oe_0_ref, t_0, hrz, n_time_rpod)

    states_roe_ws_DT_list = []
    states_rtn_ws_DT_list = []
    actions_rtn_ws_DT_list = []
    constr_DT_list = []
    constr_viol_DT_list = []
    cost_DT_list = []
    ctg_true_list = []
    for ctg_perc in np.round(np.arange(0.,1.1,0.1), decimals=1):
        ctg_true = np.round(ctg_perc*ctgs_i[0,0],2)
        print(f'CTG: {ctg_perc}; True abs:{ctgs_i[0,0]}, true ctg*abs:{ctg_true}')
        # Import the Transformer
        model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)
        DT_trajectory = DT_manager.use_model_for_imitation_learning(model, test_loader, test_sample, state_representation, rtg_perc=1., ctg_perc=ctg_perc, rtg=None, use_dynamics=True)
        states_roe_ws_DT = DT_trajectory['roe_' + transformer_ws]# set warm start
        states_rtn_ws_DT = DT_trajectory['rtn_' + transformer_ws]
        actions_rtn_ws_DT = DT_trajectory['dv_' + transformer_ws]
        cost_DT = la.norm(actions_rtn_ws_DT, axis=0).sum()
        print('ART cost:', cost_DT)
        constr_DT, constr_viol_DT = check_koz_constraint(states_rtn_ws_DT, n_time_rpod)
        print('Constr. Violation:', constr_viol_DT.sum())
        states_roe_ws_DT_list.append(states_roe_ws_DT)
        states_rtn_ws_DT_list.append(states_rtn_ws_DT)
        actions_rtn_ws_DT_list.append(actions_rtn_ws_DT)
        constr_DT_list.append(constr_DT)
        constr_viol_DT_list.append(constr_viol_DT.sum())
        cost_DT_list.append(cost_DT)
        ctg_true_list.append(ctg_true)

    results_dict['states_roe_ws_DT'][idx] = states_roe_ws_DT_list
    results_dict['states_rtn_ws_DT'][idx] = states_rtn_ws_DT_list
    results_dict['actions_rtn_ws_DT'][idx] = actions_rtn_ws_DT_list
    results_dict['constr_DT'][idx] = constr_DT_list
    results_dict['constr_viol_DT'][idx] = constr_viol_DT_list
    results_dict['cost_DT'][idx] = cost_DT_list
    results_dict['ctg_true'][idx] = ctg_true_list

np.save('results_dict_rtgcvx_ctganalysis', results_dict)