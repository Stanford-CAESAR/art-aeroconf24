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

# Initializations
warmstart = 'both' # 'cvx'/'transformer'/'both'
scenario_test_dataset = True
state_representation = 'rtn' # 'roe'/'rtn'
dataset_to_use = 'both' # 'scp'/'cvx'/'both'
transformer_ws = 'dyn' # 'dyn'/'ol'
transformer_model_name = 'checkpoint_rtn_art'
select_idx = True # set to True to manually select a test trajectory via its index (idx)
idx = 18111 # index of the test trajectory (e.g., idx = 18111)
exclude_scp_cvx = False
exclude_scp_DT = False

# Scenario sampling 
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
    if state_representation == 'roe':
        state_roe_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
    elif state_representation == 'rtn':
        state_rtn_0 = np.array((states_i[0, 0, :] * data_stats['states_std'][0]) + data_stats['states_mean'][0])
        state_roe_0 = map_rtn_to_roe(state_rtn_0, np.array(oe[0, :, 0]))
    # relativeorbit_0 = roe_to_relativeorbit(state_roe_0, oe_0_ref)

# Dynamics Matrices Precomputations
stm_hrz, cim_hrz, psi_hrz, oe_hrz, time_hrz, dt_hrz = dynamics_roe_optimization(oe_0_ref, t_0, hrz, n_time_rpod)

# Build the oe vector including the target instant
oe_hrz_trg = np.append(oe_hrz,np.array([oe_0_ref.item(0), oe_0_ref.item(1), oe_0_ref.item(2), oe_0_ref.item(3), oe_0_ref.item(4), oe_0_ref.item(5) + n_ref*(time_hrz[-1]+dt_hrz-t_0)]).reshape((6,1)),1)
time_hrz_trg = np.append(time_hrz, time_hrz[-1]+dt_hrz)

# Warmstarting and optimization
if warmstart == 'cvx' or warmstart == 'both':
    # Solve Convex Problem
    runtime_cvx0 = time.time()
    states_roe_cvx, actions_cvx, feas_cvx = ocp_cvx(stm_hrz, cim_hrz, psi_hrz, state_roe_0, n_time_rpod)
    runtime_cvx = time.time() - runtime_cvx0
    print('CVX cost:', la.norm(actions_cvx, axis=0).sum())
    print('CVX runtime:', runtime_cvx)
    states_roe_cvx_trg = np.append(states_roe_cvx, (states_roe_cvx[:,-1]+cim_hrz[:,:,-1].dot(actions_cvx[:,-1])).reshape((6,1)), 1)
    states_roe_ws_cvx = states_roe_cvx # set warm start
    states_rtn_ws_cvx = roe_to_rtn_horizon(states_roe_cvx_trg, oe_hrz_trg, n_time_rpod+1)
    # Evaluate Constraint Violation
    constr_cvx, constr_viol_cvx = check_koz_constraint(states_rtn_ws_cvx, n_time_rpod+1)
    # Solve SCP
    states_roe_scp_cvx, actions_scp_cvx, feas_scp_cvx, iter_scp_cvx , J_vect_scp_cvx, runtime_scp_cvx = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_cvx, n_time_rpod)
    if states_roe_scp_cvx is None:
        exclude_scp_cvx = True
        print('No scp-cvx solution!')
    else:
        print('SCP cost:', la.norm(actions_scp_cvx, axis=0).sum())
        print('J vect', J_vect_scp_cvx)
        print('SCP runtime:', runtime_scp_cvx)
        print('CVX+SCP runtime:', runtime_cvx+runtime_scp_cvx)
        states_roe_scp_cvx_trg = np.append(states_roe_scp_cvx, (states_roe_scp_cvx[:,-1]+cim_hrz[:,:,-1].dot(actions_scp_cvx[:,-1])).reshape((6,1)), 1)
        states_rtn_scp_cvx = roe_to_rtn_horizon(states_roe_scp_cvx_trg, oe_hrz_trg, n_time_rpod+1)
        constr_scp_cvx, constr_viol_scp_cvx = check_koz_constraint(states_rtn_scp_cvx, n_time_rpod+1)

if warmstart == 'transformer' or warmstart == 'both':
    
    # Import the Transformer
    model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)
    inference_func = getattr(DT_manager, 'torch_model_inference_'+transformer_ws)
    print('Using ART model \'', transformer_model_name, '\' with inference function DT_manage.'+inference_func.__name__+'()')
    rtg = la.norm(actions_cvx, axis=0).sum()
    DT_trajectory, runtime_DT = inference_func(model, test_loader, test_sample, stm_hrz, cim_hrz, psi_hrz, state_representation, rtg_perc=1., ctg_perc=0., rtg=rtg)
    states_roe_ws_DT = DT_trajectory['roe_' + transformer_ws]# set warm start
    # states_rtn_ws_DT = DT_trajectory['rtn_' + transformer_ws]
    actions_rtn_ws_DT = DT_trajectory['dv_' + transformer_ws]
    states_roe_DT_trg = np.append(states_roe_ws_DT, (states_roe_ws_DT[:,-1]+cim_hrz[:,:,-1].dot(actions_rtn_ws_DT[:,-1])).reshape((6,1)), 1)
    states_rtn_ws_DT = roe_to_rtn_horizon(states_roe_DT_trg, oe_hrz_trg, n_time_rpod+1)
    print('ART cost:', la.norm(actions_rtn_ws_DT, axis=0).sum())
    print('ART runtime:', runtime_DT)
    constr_DT, constr_viol_DT = check_koz_constraint(states_rtn_ws_DT, n_time_rpod+1)

    # Solve SCP
    states_roe_scp_DT, actions_scp_DT, feas_scp_DT, iter_scp_DT, J_vect_scp_DT, runtime_scp_DT = solve_scp(stm_hrz, cim_hrz, psi_hrz, state_roe_0, states_roe_ws_DT, n_time_rpod)
    if states_roe_scp_DT is None:
        exclude_scp_DT = True
        print('No scp-DT solution!')
    else:
        print('SCP cost:', la.norm(actions_scp_DT, axis=0).sum())
        print('J vect', J_vect_scp_DT)
        states_roe_scp_DT_trg = np.append(states_roe_scp_DT, (states_roe_scp_DT[:,-1]+cim_hrz[:,:,-1].dot(actions_scp_DT[:,-1])).reshape((6,1)), 1)
        states_rtn_scp_DT = roe_to_rtn_horizon(states_roe_scp_DT_trg, oe_hrz_trg, n_time_rpod+1)
        constr_scp_DT, constr_viol_scp_DT = check_koz_constraint(states_rtn_scp_DT, n_time_rpod+1)

# Plotting
plt.style.use('seaborn-v0_8-colorblind')
relativeorbit_0 = roe_to_relativeorbit(state_roe_0, oe_0_ref)
t_ws_show = dock_wyp_sample

# 3D position trajectory'
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(projection='3d')
ax1.view_init(elev=15, azim=-60, roll=0)
if warmstart == 'cvx' or warmstart == 'both':
    p1 = ax1.plot3D(states_rtn_ws_cvx[1,:t_ws_show], states_rtn_ws_cvx[2,:t_ws_show], states_rtn_ws_cvx[0,:t_ws_show], 'k--', linewidth=2.5, label='CVX')
    if not exclude_scp_cvx:
        p2 = ax1.plot3D(states_rtn_scp_cvx[1,:], states_rtn_scp_cvx[2,:], states_rtn_scp_cvx[0,:], 'k-', linewidth=3, label='SCP-CVX') # 'scp (cvx)_(' + str(iter_scp_cvx) + ')'
if warmstart == 'transformer' or warmstart == 'both':
    p3 = ax1.plot3D(states_rtn_ws_DT[1,:t_ws_show], states_rtn_ws_DT[2,:t_ws_show], states_rtn_ws_DT[0,:t_ws_show], 'b--', linewidth=2.5, label='ART') # 'warm-start ART-' + transformer_ws
    if not exclude_scp_DT:
        p4 = ax1.plot3D(states_rtn_scp_DT[1,:], states_rtn_scp_DT[2,:], states_rtn_scp_DT[0,:], 'b-', linewidth=3, label='SCP-ART') #scp (ART-' + transformer_ws + ')_(' + str(iter_scp_DT) + ')
pwyp = ax1.scatter(dock_wyp[1], dock_wyp[2], dock_wyp[0], color = 'r', marker = '*', linewidth=2.5, label='way-point')
pell = ax1.plot_surface(y_ell, z_ell, x_ell, rstride=1, cstride=1, color='r', linewidth=0, alpha=0.3, label='keep-out-zone')
pell._facecolors2d = pell._facecolor3d
pell._edgecolors2d = pell._edgecolor3d
pcone = ax1.plot_surface(t_cone, n_cone, r_cone, rstride=1, cstride=1, color='g', linewidth=0, alpha=0.7, label='approach cone')
pcone._facecolors2d = pcone._facecolor3d
pcone._edgecolors2d = pcone._edgecolor3d
p3 = ax1.plot3D(relativeorbit_0[1,:], relativeorbit_0[2,:], relativeorbit_0[0,:], '-.', color='gray', linewidth=1.5, label='initial rel. orbit')
#if not exclude_scp_cvx:
p4 = ax1.scatter(states_rtn_scp_cvx[1,0], states_rtn_scp_cvx[2,0], states_rtn_scp_cvx[0,0], color = 'b', marker = 'o', linewidth=1.5, label='$t_0$')
p5 = ax1.scatter(states_rtn_scp_cvx[1,-1], states_rtn_scp_cvx[2,-1], states_rtn_scp_cvx[0,-1], color = 'g', marker = '*', linewidth=1.5, label='docking port')
ax1.set_xlabel('\n$\delta r_t$ [m]', fontsize=15, linespacing=1.5)
ax1.set_ylabel('$\delta r_n$ [m]', fontsize=15)
ax1.set_zlabel('$\delta r_r$ [m]', fontsize=15)
ax1.tick_params(axis='y', labelcolor='k', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='z', labelsize=15)
ax1.set_xticks(np.linspace(-200, 100, 4))
ax1.set_yticks(np.linspace(-100, 100, 3))
# ax.grid(True)
#ax1.legend(loc='upper left')
ax1.set_box_aspect(aspect=None, zoom=0.9)
plt.tight_layout()
# plt.subplots_adjust(wspace=0.05)
handles1, labels1 = ax1.get_legend_handles_labels()
first_legend = plt.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, 0.85),
          ncol=4, fancybox=True, shadow=True, fontsize=15)
plt.savefig(root_folder + '/optimization/saved_files/plots/pos_3d.png', dpi = 600, bbox_inches='tight')


# 3D position trajectory'
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.view_init(elev=6, azim=-64, roll=0)
if warmstart == 'cvx' or warmstart == 'both':
    p1 = ax1.plot3D(states_rtn_ws_cvx[1,:t_ws_show], states_rtn_ws_cvx[2,:t_ws_show], states_rtn_ws_cvx[0,:t_ws_show], 'k--', linewidth=2.5, label='CVX')
    if not exclude_scp_cvx:
        p2 = ax1.plot3D(states_rtn_scp_cvx[1,:], states_rtn_scp_cvx[2,:], states_rtn_scp_cvx[0,:], 'k-', linewidth=0) # 'scp (cvx)_(' + str(iter_scp_cvx) + ')'
if warmstart == 'transformer' or warmstart == 'both':
    p3 = ax1.plot3D(states_rtn_ws_DT[1,:t_ws_show], states_rtn_ws_DT[2,:t_ws_show], states_rtn_ws_DT[0,:t_ws_show], 'b--', linewidth=2.5, label='ART') # 'warm-start ART-' + transformer_ws
    if not exclude_scp_DT:
        p4 = ax1.plot3D(states_rtn_scp_DT[1,:], states_rtn_scp_DT[2,:], states_rtn_scp_DT[0,:], 'b-', linewidth=0) #scp (ART-' + transformer_ws + ')_(' + str(iter_scp_DT) + ')
pwyp = ax1.scatter(dock_wyp[1], dock_wyp[2], dock_wyp[0], color = 'r', marker = '*', linewidth=2.5)
pell = ax1.plot_surface(y_ell, z_ell, x_ell, rstride=1, cstride=1, color='r', linewidth=0, alpha=0.3)
pell._facecolors2d = pell._facecolor3d
pell._edgecolors2d = pell._edgecolor3d
pcone = ax1.plot_surface(t_cone, n_cone, r_cone, rstride=1, cstride=1, color='g', linewidth=0, alpha=0.7)
pcone._facecolors2d = pcone._facecolor3d
pcone._edgecolors2d = pcone._edgecolor3d
p3 = ax1.plot3D(relativeorbit_0[1,:], relativeorbit_0[2,:], relativeorbit_0[0,:], '-.', color='gray', linewidth=1.5)
#if not exclude_scp_cvx:
p4 = ax1.scatter(states_rtn_scp_cvx[1,0], states_rtn_scp_cvx[2,0], states_rtn_scp_cvx[0,0], color = 'b', marker = 'o', linewidth=1.5)
p5 = ax1.scatter(states_rtn_scp_cvx[1,-1], states_rtn_scp_cvx[2,-1], states_rtn_scp_cvx[0,-1], color = 'g', marker = '*', linewidth=1.5)
ax1.set_xlabel('\n$\delta r_t$ [m]', fontsize=15, linespacing=1.5)
ax1.set_ylabel('$\delta r_n$ [m]', fontsize=15)
ax1.set_zlabel('$\delta r_r$ [m]', fontsize=15)
ax1.tick_params(axis='y', labelcolor='k', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='z', labelsize=15)
ax1.set_yticks(np.linspace(-100, 100, 3))
# ax1.set_xlim([-200,200])
# ax1.set_ylim([-300,300])
# ax1.set_zlim([-100,100])
ax1.set_box_aspect(aspect=None, zoom=0.945)
# ax.grid(True)
#ax1.legend(loc='upper left')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.view_init(elev=6, azim=-64, roll=0)
if warmstart == 'cvx' or warmstart == 'both':
    p1 = ax2.plot3D(states_rtn_ws_cvx[1,:t_ws_show], states_rtn_ws_cvx[2,:t_ws_show], states_rtn_ws_cvx[0,:t_ws_show], 'k-', linewidth=0)
    if not exclude_scp_cvx:
        p2 = ax2.plot3D(states_rtn_scp_cvx[1,:], states_rtn_scp_cvx[2,:], states_rtn_scp_cvx[0,:], 'k-', linewidth=2.5, label='SCP-CVX') # 'scp (cvx)_(' + str(iter_scp_cvx) + ')'
if warmstart == 'transformer' or warmstart == 'both':
    p3 = ax2.plot3D(states_rtn_ws_DT[1,:t_ws_show], states_rtn_ws_DT[2,:t_ws_show], states_rtn_ws_DT[0,:t_ws_show], 'b-', linewidth=0) # 'warm-start ART-' + transformer_ws
    if not exclude_scp_DT:
        p4 = ax2.plot3D(states_rtn_scp_DT[1,:], states_rtn_scp_DT[2,:], states_rtn_scp_DT[0,:], 'b-', linewidth=2.5, label='SCP-ART') #scp (ART-' + transformer_ws + ')_(' + str(iter_scp_DT) + ')
pwyp = ax2.scatter(dock_wyp[1], dock_wyp[2], dock_wyp[0], color = 'r', marker = '*', linewidth=1.5, label='way-point')
pell = ax2.plot_surface(y_ell, z_ell, x_ell, rstride=1, cstride=1, color='r', linewidth=0, alpha=0.3, label='keep-out-zone')
pell._facecolors2d = pell._facecolor3d
pell._edgecolors2d = pell._edgecolor3d
pcone = ax2.plot_surface(t_cone, n_cone, r_cone, rstride=1, cstride=1, color='g', linewidth=0, alpha=0.7, label='approach cone')
pcone._facecolors2d = pcone._facecolor3d
pcone._edgecolors2d = pcone._edgecolor3d
p3 = ax2.plot3D(relativeorbit_0[1,:], relativeorbit_0[2,:], relativeorbit_0[0,:], '-.', color='gray', linewidth=1.5, label='initial rel. orbit')
p4 = ax2.scatter(states_rtn_scp_cvx[1,0], states_rtn_scp_cvx[2,0], states_rtn_scp_cvx[0,0], color = 'b', marker = 'o', linewidth=2.5, label='$t_0$')
p5 = ax2.scatter(states_rtn_scp_cvx[1,-1], states_rtn_scp_cvx[2,-1], states_rtn_scp_cvx[0,-1], color = 'g', marker = '*', linewidth=2.5, label='docking port')
ax2.set_xlabel('\n$\delta r_t$ [m]', fontsize=15, linespacing=1.5)
ax2.set_ylabel('$\delta r_n$ [m]', fontsize=15)
ax2.set_zlabel('$\delta r_r$ [m]', fontsize=15)
ax2.tick_params(axis='y', labelcolor='k', labelsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='z', labelsize=15)
ax2.set_yticks(np.linspace(-100, 100, 3))
# ax2.set_xlim([-200,200])
# ax2.set_ylim([-300,300])
# ax2.set_zlim([-100,100])
ax2.set_box_aspect(aspect=None, zoom=0.945)
plt.tight_layout(pad=1.0, w_pad=0.2)
# plt.subplots_adjust(wspace=-0.01)
handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
first_legend = plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(-0., 0.85),
          ncol=5, fancybox=True, shadow=True, fontsize=15)
ax2.add_artist(first_legend)
#ax1.legend(loc='lower center', bbox_to_anchor=(0.8, 0.9),
#          ncol=4, fancybox=True, shadow=True, fontsize=8, zorder=1)
plt.savefig(root_folder + '/optimization/saved_files/plots/pos_3d_split.png', dpi = 600, bbox_inches='tight')

# Constraint satisfaction
plt.figure(figsize=(6,4))
if warmstart == 'cvx' or warmstart == 'both':
    plt.plot(time_hrz_trg[:t_ws_show]/period_ref, constr_cvx[:t_ws_show], 'k--', linewidth=1.5, label='CVX')
    if not exclude_scp_cvx:
        plt.plot(time_hrz_trg/period_ref, constr_scp_cvx, 'k-', linewidth=1.8, label='SCP-CVX')

if warmstart == 'transformer' or warmstart == 'both':
    plt.plot(time_hrz_trg[:t_ws_show]/period_ref, constr_DT[:t_ws_show], 'b--', linewidth=1.5, label='ART')
    if not exclude_scp_DT:
        plt.plot(time_hrz_trg/period_ref, constr_scp_DT, 'b-', linewidth=1.8, label='SCP-ART')

plt.plot(time_hrz_trg/period_ref, np.ones(n_time_rpod+1), 'r-', linewidth=1.8, label='koz')
plt.xlabel('time [orbits]', fontsize=15)
plt.ylabel('keep-out-zone constraint [-]', fontsize=15)
# plt.grid(True)
plt.xlim([-0.1,3])
plt.ylim([-0.1,9])
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(root_folder + '/optimization/saved_files/plots/koz_constr.png', dpi = 600)

fig, ax = plt.subplots(figsize=(6,4))
if warmstart == 'cvx' or warmstart == 'both':
    ax.plot(time_hrz_trg[:t_ws_show]/period_ref, constr_cvx[:t_ws_show], 'k--', linewidth=1.8, label='CVX')
    if not exclude_scp_cvx:
        ax.plot(time_hrz_trg/period_ref, constr_scp_cvx, 'k-', linewidth=2, label='SCP-CVX')

if warmstart == 'transformer' or warmstart == 'both':
    ax.plot(time_hrz_trg[:t_ws_show]/period_ref, constr_DT[:t_ws_show], 'b--', linewidth=1.8, label='ART')
    if not exclude_scp_DT:
        ax.plot(time_hrz_trg/period_ref, constr_scp_DT, 'b-', linewidth=2, label='SCP-ART')

ax.plot(time_hrz_trg/period_ref, np.ones(n_time_rpod+1), 'r-', linewidth=1.5, label='koz')
ax.fill_between([0, (time_hrz_trg/period_ref)[-1]], [0, 0], [1, 1], alpha=0.15, color='red')
ax.set_xlabel('Time [orbits]', fontsize=16, linespacing=1.5)
ax.set_ylabel('Keep-out-zone constraint [-]', fontsize=16)
ax.tick_params(axis='y', labelcolor='k', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
# plt.grid(True)
ax.set_xlim([-0.1,3])
ax.set_ylim([-0.1,9])
plt.legend(loc='best', fontsize=15)
plt.tight_layout()
plt.savefig(root_folder + '/optimization/saved_files/plots/koz_constr_v2.png', dpi = 600, bbox_inches='tight')

# ROE plots

# # ROE space
# plt.figure()
# if warmstart == 'cvx' or warmstart == 'both':
#     p1 = plt.plot(states_roe_cvx_trg[1,:t_ws_show], states_roe_cvx_trg[0,:t_ws_show], 'k--', linewidth=1.2, label='CVX')
#     if not exclude_scp_cvx:
#         p2 = plt.plot(states_roe_scp_cvx_trg[1, :], states_roe_scp_cvx_trg[0,:], 'k-', linewidth=1.5, label='SCP-CVX')
# if warmstart == 'transformer' or warmstart == 'both':
#     p3 = plt.plot(states_roe_DT_trg[1,:t_ws_show], states_roe_DT_trg[0,:t_ws_show], 'b--', linewidth=1.2, label='ART')
#     if not exclude_scp_DT:
#         p4 = plt.plot(states_roe_scp_DT_trg[1, :], states_roe_scp_DT_trg[0,:], 'b-', linewidth=1.5, label='SCP-ART')
# plt.xlabel('$a \delta \lambda$ [m]')
# plt.ylabel('$a \delta a$ [m]')
# # plt.grid(True)
# plt.legend(loc='best')
# plt.savefig(root_folder + '/optimization/saved_files/plots/roe12.png')

# plt.figure()
# if warmstart == 'cvx' or warmstart == 'both':
#     p1 = plt.plot(states_roe_cvx_trg[2,:t_ws_show], states_roe_cvx_trg[3,:t_ws_show], 'k--', linewidth=1.2, label='CVX')
#     if not exclude_scp_cvx:
#         p2 = plt.plot(states_roe_scp_cvx_trg[2, :], states_roe_scp_cvx_trg[3,:], 'k-', linewidth=1.5, label='SCP-CVX')
# if warmstart == 'transformer' or warmstart == 'both':
#     p3 = plt.plot(states_roe_DT_trg[2,:t_ws_show], states_roe_DT_trg[3,:t_ws_show], 'b--', linewidth=1.2, label='ART')
#     if not exclude_scp_DT:
#         p4 = plt.plot(states_roe_scp_DT_trg[2, :], states_roe_scp_DT_trg[3,:], 'b-', linewidth=1.5, label='SCP-ART')
# plt.xlabel('$a \delta e_x$ [m]')
# plt.ylabel('$a \delta e_y$ [m]')
# # plt.grid(True)
# plt.legend(loc='best')
# plt.savefig(root_folder + '/optimization/saved_files/plots/roe34.png')

# plt.figure()
# if warmstart == 'cvx' or warmstart == 'both':
#     p1 = plt.plot(states_roe_cvx_trg[4,:t_ws_show], states_roe_cvx_trg[5,:t_ws_show], 'k--', linewidth=1.2, label='CVX')
#     if not exclude_scp_cvx:
#         p2 = plt.plot(states_roe_scp_cvx_trg[4, :], states_roe_scp_cvx_trg[5,:], 'k-', linewidth=1.5, label='SCP-CVX')
# if warmstart == 'transformer' or warmstart == 'both':
#     p3 = plt.plot(states_roe_DT_trg[4,:t_ws_show], states_roe_DT_trg[5,:t_ws_show], 'b--', linewidth=1.2, label='ART')
#     if not exclude_scp_DT:
#         p4 = plt.plot(states_roe_scp_DT_trg[4, :], states_roe_scp_DT_trg[5,:], 'b-', linewidth=1.5, label='SCP-ART')
# plt.xlabel('$a \delta i_x$ [m]')
# plt.ylabel('$a \delta i_y$ [m]')
# # plt.grid(True)
# plt.legend(loc='best')
# plt.savefig(root_folder + '/optimization/saved_files/plots/roe56.png')

# ROE vs time
plot_orb_time = True
plt.figure(figsize=(12,8)) #figsize=(20,5)
for j in range(6):
    plt.subplot(3,2,j+1)
    if warmstart == 'cvx' or warmstart == 'both':
        plt.plot(time_hrz_trg[:t_ws_show]/period_ref, states_roe_cvx_trg[j,:t_ws_show], 'k--', linewidth=1.5, label='CVX')
        if not exclude_scp_cvx:
            plt.plot(time_hrz_trg/period_ref, states_roe_scp_cvx_trg[j,:], 'k-', linewidth=1.8, label='SCP-CVX')
    if warmstart == 'transformer' or warmstart == 'both':
        plt.plot(time_hrz_trg[:t_ws_show]/period_ref, states_roe_DT_trg[j,:t_ws_show], 'b--', linewidth=1.5, label='ART')
        if not exclude_scp_DT:
            plt.plot(time_hrz_trg/period_ref, states_roe_scp_DT_trg[j,:], 'b-', linewidth=1.8, label='SCP-ART')
    if j == 0:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta a$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-60,30])
        plt.legend(loc='best', fontsize=11.5)
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(-50,50,25))
    elif j == 1:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta \lambda$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-100,200])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(-100,250,50))
        # plt.legend(loc='best')
    elif j == 2:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta e_x$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-30,20])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(-25,25,10))
        # plt.legend(loc='best')
    elif j == 3:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta e_y$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-5,100])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(0,120,20))
        # plt.legend(loc='best')
    elif j == 4:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta i_x$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-6,6])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(np.arange(-5,7.5,2.5))
        # plt.legend(loc='best')
    elif j == 5:
        plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=20)
        plt.ylabel('$a \delta i_y$ [m]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-5,140])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(0,150,25))
        # plt.legend(loc='best')
plt.tight_layout()
plt.savefig(root_folder + '/optimization/saved_files/plots/roe_vs_time.png', dpi = 600, bbox_inches='tight')

# Control
plt.figure(figsize=(12,8)) #figsize=(20,5)
for j in range(3):
    plt.subplot(1,3,j+1)
    plt.stem(time_hrz[:t_ws_show]/period_ref, actions_cvx[j,:t_ws_show]*1000., 'k--', markerfmt='D', label='CVX')
    plt.stem(time_hrz/period_ref, actions_scp_cvx[j,:]*1000., 'k-', label='SCP-CVX')
    plt.stem(time_hrz[:t_ws_show]/period_ref, actions_rtn_ws_DT[j,:t_ws_show]*1000., 'b--', markerfmt='D', label='ART')
    plt.stem(time_hrz/period_ref, actions_scp_DT[j,:]*1000., 'b-', label='SCP-ART')    
    if j == 0:
        plt.xlabel('time [orbits]', fontsize=20)
        plt.ylabel('$ \Delta v_r$ [mm/s]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-15,25])
        plt.legend(loc='best', fontsize=15)
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.xticks(range(0,4,1))
    elif j == 1:
        plt.xlabel('time [orbits]', fontsize=20)
        plt.ylabel('$ \Delta v_t$ [mm/s]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-25,25])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.yticks(range(-25,30,5))
        plt.xticks(range(0,4,1))
        # plt.legend(loc='best')
    elif j == 2:
        plt.xlabel('time [orbits]', fontsize=20)
        plt.ylabel('$ \Delta v_n$ [mm/s]', fontsize=20)
        # plt.grid(True)
        plt.xlim([-0.1,3])
        plt.ylim([-60,60])
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=20)
        plt.xticks(range(0,4,1))
        # plt.legend(loc='best')
plt.tight_layout()
plt.savefig(root_folder + '/optimization/saved_files/plots/delta_v.png', dpi = 600, bbox_inches='tight')

# # Cost
# if not (exclude_scp_cvx or exclude_scp_DT):
#     plt.figure()
#     max_it = max(iter_scp_cvx, iter_scp_DT)
#     for i in range(max_it):
#         if i >= iter_scp_cvx:
#             J_vect_scp_cvx[i] = J_vect_scp_cvx[iter_scp_cvx-1]
#         elif i >= iter_scp_DT:
#             J_vect_scp_DT[i] = J_vect_scp_DT[iter_scp_DT-1]
#     if warmstart == 'cvx' or warmstart == 'both':
#         plt.plot(J_vect_scp_cvx[:max_it]*1000., 'b--', marker='o', linewidth=1.5, label='SCP-CVX')

#     if warmstart == 'transformer' or warmstart == 'both':
#         plt.plot(J_vect_scp_DT[:max_it]*1000., 'g--', marker='o', linewidth=1.5, label='SCP-ART')

#     plt.xlabel('Iterations [-]')
#     plt.ylabel('Cost [m/s]')
#     # plt.grid(True)
#     plt.legend(loc='best')
#     plt.savefig(root_folder + '/optimization/saved_files/plots/cost.png')

# io.savemat('plot_data.mat', dict(states_rtn_ws_cvx=states_rtn_ws_cvx, states_rtn_scp_cvx=states_rtn_scp_cvx, states_rtn_ws_DT=states_rtn_ws_DT, states_rtn_scp_DT=states_rtn_scp_DT, dock_wyp=dock_wyp, x_ell=x_ell, y_ell=y_ell, z_ell=z_ell, t_cone=t_cone, n_cone=n_cone, r_cone=r_cone))

plt.show()