import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import cvxpy as cp
import mosek as mk
import time

from optimization.rpod_scenario import *

def ocp_cvx_tpbvp(stm, cim, psi, s_0, n_time):

    s = cp.Variable((6, n_time))
    a = cp.Variable((3, n_time))

    solve_normalized = False

    if not solve_normalized:

        # Compute parameters
        s_f = state_roe_target

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim[:,:,-1] @ a[:,-1] == s_f]

        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value
        
    else:

        # Compute normalized parameters
        cim_n = cim*n_ref
        s_0_n = s_0/a_ref
        s_f_n = state_roe_target/a_ref

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0_n]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim_n[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim_n[:,:,-1] @ a[:,-1] == s_f_n]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.MOSEK, verbose=False)

        s_opt = s.value*a_ref
        a_opt = a.value*a_ref*n_ref

    return s_opt, a_opt, prob.status

def ocp_cvx(stm, cim, psi, s_0, n_time):

    s = cp.Variable((6, n_time))
    a = cp.Variable((3, n_time))

    solve_normalized = False

    if not solve_normalized:

        # Compute parameters
        s_f = state_roe_target
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim[:,:,-1] @ a[:,-1] == s_f]
        # Docking waypoint
        constraints += [psi[:,:,dock_wyp_sample] @ s[:,dock_wyp_sample] == dock_wyp]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi[:,:,j]))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi[:,:,j])
            b_soc_j = -dock_port
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value
        
    else:

        # Compute normalized parameters
        cim_n = cim*n_ref
        psi_norm_vect = np.array([1, 1, 1, 1/n_ref, 1/n_ref, 1/n_ref]).reshape(6,)
        s_0_n = s_0/a_ref
        s_f_n = state_roe_target/a_ref
        dock_wyp_n = np.multiply(dock_wyp, np.array([1/a_ref, 1/a_ref, 1/a_ref, 1/(a_ref*n_ref), 1/(a_ref*n_ref), 1/(a_ref*n_ref)]).reshape(6,))
        dock_port_n = dock_port/a_ref
        d_soc = -np.transpose(dock_axis).dot(dock_port_n)/np.cos(dock_cone_angle)

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0_n]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim_n[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim_n[:,:,-1] @ a[:,-1] == s_f_n]
        # Docking waypoint
        psi_wyp_n = psi[:,:,dock_wyp_sample]*psi_norm_vect[:, np.newaxis]
        constraints += [psi_wyp_n @ s[:,dock_wyp_sample] == dock_wyp_n]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            psi_j_n = psi[:,:,j]*psi_norm_vect[:, np.newaxis]
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi_j_n))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi_j_n)
            b_soc_j = -dock_port_n
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.MOSEK, verbose=False)

        s_opt = s.value*a_ref
        a_opt = a.value*a_ref*n_ref

    return s_opt, a_opt, prob.status

def ocp_scp(stm, cim, psi, s_0, s_ref, trust_region, n_time):

    s = cp.Variable((6, n_time))
    a = cp.Variable((3, n_time))

    solve_normalized = False

    if not solve_normalized:

        # Compute parameters
        s_f = state_roe_target
        d_soc = -np.transpose(dock_axis).dot(dock_port)/np.cos(dock_cone_angle)

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim[:,:,-1] @ a[:,-1] == s_f]
        # Docking waypoint
        constraints += [psi[:,:,dock_wyp_sample] @ s[:,dock_wyp_sample] == dock_wyp]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi[:,:,j]))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi[:,:,j])
            b_soc_j = -dock_port
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
        # Keep-out-zone plus trust region
        for k in range(dock_wyp_sample):
            c_koz_k = np.transpose(s_ref[:,k]).dot(np.matmul(np.transpose(psi[:,:,k]), np.matmul(DEED_koz, psi[:,:,k])))
            b_koz_k = np.sqrt(c_koz_k.dot(s_ref[:,k]))
            constraints += [c_koz_k @ s[:,k] >= b_koz_k]
            b_soc_k = -s_ref[:,k]
            constraints += [cp.SOC(trust_region, s[:,k] + b_soc_k)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        s_opt = s.value
        a_opt = a.value

    else:

        # Compute normalized parameters
        s_ref_n = s_ref/a_ref
        cim_n = cim*n_ref
        psi_norm_vect = np.array([1, 1, 1, 1/n_ref, 1/n_ref, 1/n_ref]).reshape(6,)
        s_0_n = s_0/a_ref
        s_f_n = state_roe_target/a_ref
        dock_wyp_n = np.multiply(dock_wyp, np.array([1/a_ref, 1/a_ref, 1/a_ref, 1/(a_ref*n_ref), 1/(a_ref*n_ref), 1/(a_ref*n_ref)]).reshape(6,))
        dock_port_n = dock_port/a_ref
        trust_region_n = trust_region/a_ref
        d_soc = -np.transpose(dock_axis).dot(dock_port_n)/np.cos(dock_cone_angle)
        DEED_koz_n = DEED_koz*a_ref**2

        # Compute Constraints
        constraints = []
        # Initial Condition
        constraints += [s[:,0] == s_0_n]
        # Dynamics
        constraints += [s[:,i+1] == stm[:,:,i] @ (s[:,i] + cim_n[:,:,i] @ a[:,i]) for i in range(n_time-1)]
        # Terminal Condition
        constraints += [s[:,-1] + cim_n[:,:,-1] @ a[:,-1] == s_f_n]
        # Docking waypoint
        psi_wyp_n = psi[:,:,dock_wyp_sample]*psi_norm_vect[:, np.newaxis]
        constraints += [psi_wyp_n @ s[:,dock_wyp_sample] == dock_wyp_n]
        # Approach cone
        for j in range(dock_wyp_sample, n_time):
            psi_j_n = psi[:,:,j]*psi_norm_vect[:, np.newaxis]
            c_soc_j = np.transpose(dock_axis).dot(np.matmul(D_pos, psi_j_n))/np.cos(dock_cone_angle)
            A_soc_j = np.matmul(D_pos, psi_j_n)
            b_soc_j = -dock_port_n
            constraints += [cp.SOC(c_soc_j @ s[:,j] + d_soc, A_soc_j @ s[:,j] + b_soc_j)]
        # Keep-out-zone plus trust region
        for k in range(dock_wyp_sample):
            psi_k_n = psi[:,:,k]*psi_norm_vect[:, np.newaxis]
            c_koz_k = np.transpose(s_ref_n[:,k]).dot(np.matmul(np.transpose(psi_k_n), np.matmul(DEED_koz_n, psi_k_n)))
            b_koz_k = np.sqrt(c_koz_k.dot(s_ref_n[:,k]))
            constraints += [c_koz_k @ s[:,k] >= b_koz_k]
            b_soc_k = -s_ref_n[:,k]
            constraints += [cp.SOC(trust_region_n, s[:,k] + b_soc_k)]
    
        # Compute Cost
        cost = cp.sum(cp.norm(a, 2, axis=0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve(solver=cp.MOSEK, verbose=False)

        s_opt = s.value*a_ref
        a_opt = a.value*a_ref*n_ref

    return s_opt, a_opt, prob.status, prob.value

def solve_scp(stm, cim, psi, state_roe_0, states_roe_ref, n_time):

    beta_SCP = (trust_regionf/trust_region0)**(1/iter_max_SCP)

    iter_SCP = 0
    DELTA_J = 10
    J_vect = np.ones(shape=(iter_max_SCP,), dtype=float)*1e12

    diff = trust_region0
    trust_region = trust_region0
    
    runtime0_scp = time.time()
    while (iter_SCP < iter_max_SCP) and ((diff > trust_regionf) or (DELTA_J > J_tol)):
        
        # Solve OCP (safe)
        try:
            [states_roe, actions, feas, cost] = ocp_scp(stm, cim, psi, state_roe_0, states_roe_ref, trust_region, n_time)
        except:
            states_roe = None
            actions = None
            feas = 'failure'

        if np.char.equal(feas,'optimal'):
            if iter_SCP == 0:
                states_roe_vect = states_roe[None,:,:].copy()
                actions_vect = actions[None,:,:].copy()
            else:
                states_roe_vect = np.vstack((states_roe_vect, states_roe[None,:,:]))
                actions_vect = np.vstack((actions_vect, actions[None,:,:]))

            # Compute performances
            diff = np.max(la.norm(states_roe - states_roe_ref, axis=0))
            # print('scp gap:', diff)
            J = sum(la.norm(actions,axis=0));#2,1
            J_vect[iter_SCP] = J
            #print(J)

            # Update iteration
            iter_SCP += 1
            
            if iter_SCP > 1:
               DELTA_J = J_old - J
            J_old = J
            #print('-----')
            #print(DELTA_J)
            #print('*****')

            #  Update trust region
            trust_region = beta_SCP * trust_region
            
            #  Update reference
            states_roe_ref = states_roe
        else:
            print('unfeasible scp')
            break;
    
    runtime1_scp = time.time()
    runtime_scp = runtime1_scp - runtime0_scp
    
    ind_J_min = np.argmin(J_vect)
    if np.char.equal(feas,'optimal'):
        states_roe = states_roe_vect[ind_J_min,:,:]
        actions = actions_vect[ind_J_min,:,:]
    else:
        states_roe = None
        actions = None

    return states_roe, actions, feas, iter_SCP, J_vect, runtime_scp

def check_koz_constraint(states_rtn, n_time):

    constr_koz = np.empty(shape=(n_time,), dtype=float)
    constr_koz_violation = np.zeros(shape=(n_time,), dtype=float)

    for i in range(n_time):
        constr_koz[i] = np.transpose(states_rtn[:3, i]).dot(EE_koz.dot(states_rtn[:3, i]))
        if (constr_koz[i] < 1) and (i < dock_wyp_sample):
            constr_koz_violation[i] = 1

    return constr_koz, constr_koz_violation

def compute_constraint_to_go(states_rtn, n_data, n_time):

    constraint_to_go = np.empty(shape=(n_data, n_time), dtype=float)
    for n in range(n_data):
        constr_koz_n, constr_koz_violation_n = check_koz_constraint(np.transpose(np.squeeze(states_rtn[n, :, :])), n_time)
        for t in range(n_time):
            constraint_to_go[n, t] = np.sum(constr_koz_violation_n[t:])

    return constraint_to_go

def compute_reward_to_go(actions, n_data, n_time):

    rewards_to_go = np.empty(shape=(n_data, n_time), dtype=float)
    for n in range(n_data):
        for t in range(n_time):
            rewards_to_go[n, t] = - np.sum(la.norm(actions[n, t:, :], axis=1))
        
    return rewards_to_go