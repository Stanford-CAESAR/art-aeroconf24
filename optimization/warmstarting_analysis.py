import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(root_folder)
print(sys.path)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt

from dynamics.orbit_dynamics import *
from rpod_scenario import *
from ocp import *
import transformer.manage as DT_manager
import itertools
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

def for_computation(input_iterable):

    # Extract input
    current_idx = input_iterable[0]
    input_dict = input_iterable[1]
    model = input_dict['model']
    test_loader = input_dict['test_loader']
    state_representation = input_dict['state_representation']
    transformer_ws = input_dict['transformer_ws']

    # Output dictionary initialization
    out = {'feasible_cvx' : True,
           'feasible_scp_cvx' : True,
           'feasible_DT' : True,
           'J_vect_scp_cvx': [],
           'J_vect_scp_DT': [],
           'J_cvx' : [],
           'J_DT' : [],
           'iter_scp_cvx': [],
           'iter_scp_DT': [],
           'runtime_cvx': [],
           'runtime_DT': [],
           'runtime_scp_cvx': [],
           'runtime_scp_DT': [],
           'ctgs0_cvx': [],
           'cvx_problem' : False,
           'test_dataset_ix' : [],
           'feasible_cvx_tpbvp' : True,
           'feasible_scp_cvx_tpbvp' : True,
           'J_vect_scp_cvx_tpbvp': [],
           'J_cvx_tpbvp' : [],
           'iter_scp_cvx_tpbvp': [],
           'runtime_cvx_tpbvp': [],
           'runtime_scp_cvx_tpbvp': []
          }
   
    test_sample = test_loader.dataset.getix(current_idx)
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = test_sample

    # print('Sampled trajectory ' + str(ix) + ' from test_dataset.')
    data_stats = test_loader.dataset.data_stats
    out['test_dataset_ix'] = ix[0]

    hrz = horizons.item()
    if state_representation == 'roe':
        state_roe_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
    elif state_representation == 'rtn':
        state_rtn_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        state_roe_0 = map_rtn_to_roe(state_rtn_0, np.array(oe[0, :, 0]))

    # Dynamics Matrices Precomputations
    stm_hrz, cim_hrz, psi_hrz, oe_hrz, time_hrz, dt_hrz = dynamics_roe_optimization(oe_0_ref, t_0, hrz, n_time_rpod)

    ####### Warmstart Convex Problem TPBVP
    try:
        runtime0_cvx_tpbvp = time.time()
        states_roe_cvx_tpbvp, actions_cvx_tpbvp, feas_cvx_tpbvp = ocp_cvx_tpbvp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, n_time_rpod)
        runtime1_cvx_tpbvp = time.time()
        runtime_cvx_tpbvp = runtime1_cvx_tpbvp-runtime0_cvx_tpbvp
    except:
        states_roe_cvx_tpbvp = None
        actions_cvx_tpbvp = None
        feas_cvx_tpbvp = 'failure'
        runtime_cvx_tpbvp = None
    
    if np.char.equal(feas_cvx_tpbvp,'optimal'):
        states_roe_ws_cvx_tpbvp = states_roe_cvx_tpbvp # set warm start
        out['J_cvx_tpbvp'] = sum(la.norm(actions_cvx_tpbvp,axis=0))
        out['runtime_cvx_tpbvp'] = runtime_cvx_tpbvp

        # Solve SCP
        states_roe_scp_cvx_tpbvp, actions_scp_cvx_tpbvp, feas_scp_cvx_tpbvp, iter_scp_cvx_tpbvp, J_vect_scp_cvx_tpbvp, runtime_scp_cvx_tpbvp = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_cvx_tpbvp, n_time_rpod)
        
        if np.char.equal(feas_scp_cvx_tpbvp,'optimal'):
            # Save scp_cvx data in the output dictionary
            out['J_vect_scp_cvx_tpbvp'] = J_vect_scp_cvx_tpbvp
            out['iter_scp_cvx_tpbvp'] = iter_scp_cvx_tpbvp    
            out['runtime_scp_cvx_tpbvp'] = runtime_scp_cvx_tpbvp
        else:
            out['feasible_scp_cvx_tpbvp'] = False
    else:
        out['feasible_scp_cvx_tpbvp'] = False
        out['feasible_cvx_tpbvp'] = False

    ####### Warmstart Convex Problem RPOD
    try:
        runtime0_cvx = time.time()
        states_roe_cvx, actions_cvx, feas_cvx = ocp_cvx(stm_hrz, cim_hrz, psi_hrz, state_roe_0, n_time_rpod)
        runtime1_cvx = time.time()
        runtime_cvx = runtime1_cvx-runtime0_cvx
    except:
        states_roe_cvx = None
        actions_cvx = None
        feas_cvx = 'failure'
        runtime_cvx = None
    
    if np.char.equal(feas_cvx,'optimal'):
        states_roe_ws_cvx = states_roe_cvx # set warm start
        out['J_cvx'] = sum(la.norm(actions_cvx,axis=0))
        states_rtn_ws_cvx = roe_to_rtn_horizon(states_roe_cvx, oe_hrz, n_time_rpod)
        # Evaluate Constraint Violation
        ctgs_cvx = compute_constraint_to_go(states_rtn_ws_cvx.T[None,:,:], 1, n_time_rpod)
        ctgs0_cvx = ctgs_cvx[0,0]
        # Save cvx in the output dictionary
        out['runtime_cvx'] = runtime_cvx
        out['ctgs0_cvx'] = ctgs0_cvx
        out['cvx_problem'] = ctgs0_cvx == 0

        # Solve SCP
        states_roe_scp_cvx, actions_scp_cvx, feas_scp_cvx, iter_scp_cvx , J_vect_scp_cvx, runtime_scp_cvx = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_cvx, n_time_rpod)
        
        if np.char.equal(feas_scp_cvx,'optimal'):
            # Save scp_cvx data in the output dictionary
            out['J_vect_scp_cvx'] = J_vect_scp_cvx
            out['iter_scp_cvx'] = iter_scp_cvx    
            out['runtime_scp_cvx'] = runtime_scp_cvx
        else:
            out['feasible_scp_cvx'] = False
    else:
        out['feasible_scp_cvx'] = False
        out['feasible_cvx'] = False

    ####### Warmstart Transformer
    # Import the Transformer
    if np.char.equal(feas_cvx,'optimal'):
        rtg_0 = -out['J_cvx']
        if transformer_ws == 'dyn':
            DT_trajectory, runtime_DT = DT_manager.torch_model_inference_dyn(model, test_loader, test_sample, stm_hrz, cim_hrz, psi_hrz, state_representation, rtg_perc=None, ctg_perc=0., rtg=rtg_0)
        elif transformer_ws == 'ol':
            DT_trajectory, runtime_DT = DT_manager.torch_model_inference_ol(model, test_loader, test_sample, stm_hrz, cim_hrz, psi_hrz, state_representation, rtg_perc=None, ctg_perc=0., rtg=rtg_0)
    out['J_DT'] = sum(la.norm(DT_trajectory['dv_' + transformer_ws],axis=0))
    states_roe_ws_DT = DT_trajectory['roe_' + transformer_ws] # set warm start
    # Save DT in the output dictionary
    out['runtime_DT'] = runtime_DT

    # Solve SCP
    states_roe_scp_DT, actions_scp_DT, feas_scp_DT, iter_scp_DT, J_vect_scp_DT, runtime_scp_DT = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_DT, n_time_rpod)
    
    if np.char.equal(feas_scp_DT,'optimal'):
        # Save scp_DT in the output dictionary
        out['J_vect_scp_DT'] = J_vect_scp_DT
        out['iter_scp_DT'] = iter_scp_DT
        out['runtime_scp_DT'] = runtime_scp_DT
    else:
        out['feasible_DT'] = False   

    return out

if __name__ == '__main__':

    state_representation = 'rtn' # 'roe'/'rtn'
    dataset_to_use = 'scp' # 'scp'/'cvx'/'both'
    transformer_ws = 'dyn' # 'dyn'/'ol'
    transformer_model_name = 'checkpoint_rtn_v11' # 'checkpoint_rtn_v11'/'checkpoint_rtn_v01'/'checkpoint_rtn_ctgrtg'/checkpoint_rtn_ctgrtg_v02
    set_start_method('spawn')
    num_processes = 8

    # Get the datasets and loaders from the torch data
    datasets, dataloaders = DT_manager.get_train_val_test_data(state_representation, dataset_to_use, transformer_model_name)
    train_loader, eval_loader, test_loader = dataloaders
    model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)

    # Parallel for inputs
    N_data_test = test_loader.dataset.n_data
    other_args = {
        'model' : model,
        'test_loader' : test_loader,
        'state_representation' : state_representation,
        'transformer_ws' : transformer_ws
    }

    J_vect_scp_cvx = np.empty(shape=(N_data_test, iter_max_SCP), dtype=float) 
    J_vect_scp_DT = np.empty(shape=(N_data_test, iter_max_SCP), dtype=float)
    J_cvx = np.empty(shape=(N_data_test, ), dtype=float)
    J_DT = np.empty(shape=(N_data_test, ), dtype=float)
    iter_scp_cvx = np.empty(shape=(N_data_test, ), dtype=float) 
    iter_scp_DT = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_cvx = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_DT = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_scp_cvx = np.empty(shape=(N_data_test, ), dtype=float) 
    runtime_scp_DT = np.empty(shape=(N_data_test, ), dtype=float) 
    ctgs0_cvx = np.empty(shape=(N_data_test, ), dtype=float)
    cvx_problem = np.full(shape=(N_data_test, ), fill_value=False)
    test_dataset_ix = np.empty(shape=(N_data_test, ), dtype=float)
    J_vect_scp_cvx_tpbvp = np.empty(shape=(N_data_test, iter_max_SCP), dtype=float) 
    J_cvx_tpbvp = np.empty(shape=(N_data_test, ), dtype=float)
    iter_scp_cvx_tpbvp = np.empty(shape=(N_data_test, ), dtype=float)
    runtime_cvx_tpbvp = np.empty(shape=(N_data_test, ), dtype=float)
    runtime_scp_cvx_tpbvp = np.empty(shape=(N_data_test, ), dtype=float)

    i_unfeas_cvx = []
    i_unfeas_scp_cvx = []
    i_unfeas_DT = []
    i_unfeas_cvx_tpbvp = []
    i_unfeas_scp_cvx_tpbvp = []

    # Pool creation --> Should automatically select the maximum number of processes
    p = Pool(processes=num_processes)
    for i, res in enumerate(tqdm(p.imap(for_computation, zip(np.arange(N_data_test), itertools.repeat(other_args))), total=N_data_test)):
        # Save the input in the dataset
        test_dataset_ix[i] = res['test_dataset_ix']

        # If the solution is feasible save the optimization output
        if res['feasible_cvx']:
            J_cvx[i] = res['J_cvx']
            runtime_cvx[i] = res['runtime_cvx']
            ctgs0_cvx[i] = res['ctgs0_cvx']
            cvx_problem[i] = res['cvx_problem']
        else:
            i_unfeas_cvx += [ i ]

        if res['feasible_scp_cvx']:
            J_vect_scp_cvx[i,:] = res['J_vect_scp_cvx']
            iter_scp_cvx[i] = res['iter_scp_cvx']
            runtime_scp_cvx[i] = res['runtime_scp_cvx']
        else:
            i_unfeas_scp_cvx += [ i ]

        if res['feasible_DT']:
            J_DT[i] = res['J_DT']
            J_vect_scp_DT[i,:] = res['J_vect_scp_DT']
            iter_scp_DT[i] = res['iter_scp_DT']
            runtime_DT[i] = res['runtime_DT']
            runtime_scp_DT[i] = res['runtime_scp_DT']
        else:
            i_unfeas_DT += [ i ]

        if res['feasible_cvx_tpbvp']:
            J_cvx_tpbvp[i] = res['J_cvx_tpbvp']
            runtime_cvx_tpbvp[i] = res['runtime_cvx_tpbvp']
        else:
            i_unfeas_cvx_tpbvp += [ i ]

        if res['feasible_scp_cvx_tpbvp']:
            J_vect_scp_cvx_tpbvp[i,:] = res['J_vect_scp_cvx_tpbvp']
            iter_scp_cvx_tpbvp[i] = res['iter_scp_cvx_tpbvp']
            runtime_scp_cvx_tpbvp[i] = res['runtime_scp_cvx_tpbvp']
        else:
            i_unfeas_scp_cvx_tpbvp += [ i ]
    
    #  Save dataset (local folder for the workstation)
    np.savez_compressed(root_folder + '/optimization/saved_files/warmstarting/ws_analysis_' + transformer_model_name + '_' + transformer_ws,
                        J_vect_scp_cvx = J_vect_scp_cvx,
                        J_vect_scp_DT = J_vect_scp_DT,
                        J_cvx = J_cvx,
                        J_DT = J_DT,
                        iter_scp_cvx = iter_scp_cvx,
                        iter_scp_DT = iter_scp_DT,
                        runtime_cvx = runtime_cvx,
                        runtime_DT = runtime_DT,
                        runtime_scp_cvx = runtime_scp_cvx,
                        runtime_scp_DT = runtime_scp_DT,
                        ctgs0_cvx = ctgs0_cvx, 
                        cvx_problem = cvx_problem,
                        test_dataset_ix = test_dataset_ix,
                        i_unfeas_cvx = i_unfeas_cvx,
                        i_unfeas_scp_cvx = i_unfeas_scp_cvx,
                        i_unfeas_DT = i_unfeas_DT,
                        J_vect_scp_cvx_tpbvp = J_vect_scp_cvx_tpbvp,
                        J_cvx_tpbvp = J_cvx_tpbvp,
                        iter_scp_cvx_tpbvp = iter_scp_cvx_tpbvp,
                        runtime_cvx_tpbvp = runtime_cvx_tpbvp,
                        runtime_scp_cvx_tpbvp = runtime_scp_cvx_tpbvp,
                        i_unfeas_cvx_tpbvp = i_unfeas_cvx_tpbvp,
                        i_unfeas_scp_cvx_tpbvp = i_unfeas_scp_cvx_tpbvp
                        )