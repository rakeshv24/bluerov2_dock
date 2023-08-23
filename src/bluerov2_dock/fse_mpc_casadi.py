from copy import deepcopy

import numpy as np
import numpy.linalg as LA
import yaml
from casadi import (DM, SX, Function, Opti, cumsum, det, eig_symbolic, evalf,
                    gradient, integrator, inv, matrix_expand, mmin, mtimes,
                    skew, sum1, vertcat)
from scipy.linalg import block_diag
from scipy.optimize import minimize

from auv_hinsdale import AUV


class FEMPC(object):
    def __init__(self, auv, fse_mpc_params):
        self.auv = auv
        self.dt = fse_mpc_params["dt"]
        self.log_quiet = fse_mpc_params["quiet"]
        self.horizon = fse_mpc_params["horizon"]
        self.thrusters = fse_mpc_params["thrusters"]
        self.window_size = fse_mpc_params["window_size"]
        self.dt_int = str(fse_mpc_params["dt_int"])
        self.model_type = fse_mpc_params["full_body"]
        self.est_flag = fse_mpc_params["estimation"]
        
        self.Q_eta = float(fse_mpc_params["noises"]["Q_eta"]) * np.eye(6)
        self.Q_nu_r = float(fse_mpc_params["noises"]["Q_nu_r"]) * np.eye(6)
        self.Q_f = float(fse_mpc_params["noises"]["Q_f"]) * np.eye(3)
        self.Q_auv_state = block_diag(self.Q_eta, self.Q_nu_r)
        
        self.R_att = float(fse_mpc_params["noises"]["R_att"]) * np.eye(3)
        self.R_linvel = float(fse_mpc_params["noises"]["R_linvel"]) * np.eye(3)
        self.R_angvel = float(fse_mpc_params["noises"]["R_angvel"]) * np.eye(3)
        self.R_linacc = float(fse_mpc_params["noises"]["R_linacc"]) * np.eye(3)
        self.R_xy = float(fse_mpc_params["noises"]["R_xy"]) * np.eye(2)
        self.R_z = float(fse_mpc_params["noises"]["R_z"])
        self.R_dr = float(fse_mpc_params["noises"]["R_dr"])
        self.R_auv_meas = block_diag(self.R_z, self.R_att, self.R_angvel, self.R_linacc)
        self.R_f_meas = block_diag(self.R_linacc, self.R_dr)

        self.P = np.diag(fse_mpc_params["penalty"]["P"])
        self.Q = np.diag(fse_mpc_params["penalty"]["Q"])
        self.R = np.diag(fse_mpc_params["penalty"]["R"])

        self.xmin = np.array(fse_mpc_params["bounds"]["xmin"], dtype=np.float64).reshape(-1, 1)
        self.xmax = np.array(fse_mpc_params["bounds"]["xmax"], dtype=np.float64).reshape(-1, 1)
        self.umin = np.array(fse_mpc_params["bounds"]["umin"], dtype=np.float64).reshape(self.thrusters, 1)
        self.umax = np.array(fse_mpc_params["bounds"]["umax"], dtype=np.float64).reshape(self.thrusters, 1)
        self.dumin = np.array(fse_mpc_params["bounds"]["dumin"], dtype=np.float64).reshape(self.thrusters, 1)
        self.dumax = np.array(fse_mpc_params["bounds"]["dumax"], dtype=np.float64).reshape(self.thrusters, 1)
        
        self.wec_states = None

        self.reset()
        self.initialize_optimizer()

    @classmethod
    def load_params(cls, auv_filename, mpc_filename):
        auv = AUV.load_params(auv_filename)

        f = open(mpc_filename, "r")
        fse_mpc_params = yaml.load(f.read(), Loader=yaml.SafeLoader)
        
        return cls(auv, fse_mpc_params)
    
    def reset(self):
        self.previous_control = None
        self.previous_state = None

    def initialize_optimizer(self):
        chi = SX.sym('chi', (18,1))
        chi_next = SX.sym('chi_next', (18,1))
        u = SX.sym('u', (self.thrusters,1))
        f_B = SX.sym('f_B', (3,1))
        nu_w = SX.sym('nu_w', (3,1))
        nu_w_dot = SX.sym('nu_w_dot', (3,1))
        full_body = SX.sym('full_body')
        f_est = SX.sym('f_est')
        f_I = SX.sym('f_I', (3,1))
        eta = SX.sym('eta', (6,1))
        
        flow_body = self.flow_RI2B(f_I, eta)
        self.get_flow = Function('f_ned', [f_I, eta], [flow_body])
        
        chi_k_1 = self.forward_dyn(chi, u, f_B, nu_w, nu_w_dot, f_est, full_body)
        self.forward_dynamics = Function('f',[chi, u, f_B, nu_w, nu_w_dot, f_est, full_body],[chi_k_1])
        
        H_kk_next = self.calc_meas_H(chi_next)
        self.calc_H_kk_next = Function('f1', [chi_next], [H_kk_next])
        
    def flow_RI2B(self, f_I, eta):
        tf_B2I = self.auv.compute_transformation_matrix(eta)
        R_B2I = tf_B2I[0:3, 0:3]
        f_B = R_B2I @ f_I
        return f_B            
        
    def forward_dyn(self, chi, u, f_B, nu_w, nu_w_dot, f_est, full_body):
        chi_dot = self.auv.compute_nonlinear_dynamics(chi, u, f_B=f_B, nu_w=nu_w, nu_w_dot=nu_w_dot, f_est=f_est, complete_model=full_body)
        chi_k_1 = chi + (chi_dot * self.dt)
        return chi_k_1
        
    def calc_meas_H(self, x_next):
        S_kk_next = skew(x_next[9:12, :])
        tf_B2I_est = self.auv.compute_transformation_matrix(x_next[0:6, :])
        R_B2I_est = tf_B2I_est[0:3, 0:3]
        H_kk_next = vertcat(-S_kk_next, SX([[0, 0, 1]]) @ R_B2I_est)
        return H_kk_next
    
    def cost_fn(self, u, chi, f_B, ctrl_obj):
        chi_dot = self.auv.compute_nonlinear_dynamics(chi, u, f_B=f_B, f_est=self.est_flag, complete_model=self.model_type)
        chi_next = chi + chi_dot[0:12] * self.dt
        
        S_kk_next = evalf(skew(chi_next[9:12, :]))
        tf_B2I_est = self.auv.compute_transformation_matrix(chi_next[0:6, :])
        R_B2I_est = evalf(tf_B2I_est[0:3, 0:3])
        H_kk_next = np.vstack((-S_kk_next, np.array([[0, 0, 1]]) @ R_B2I_est))
        
        gram = (H_kk_next.T @ np.linalg.inv(self.R_f_meas) @ H_kk_next + ctrl_obj["G"])
        # cost = np.mean(np.diag(np.cov(gram)))
        mean_var = np.mean(np.diag(np.cov(gram)))
        max_var = np.max(np.diag(np.cov(gram)))
        min_eig = -np.min(LA.eigvals(gram))
        cost = mean_var
        return cost   
    
    def ap_control_cg(self, idx, chi, f_B, ctrl_obj):
        eta = chi[0:6, :]
        nu_r = chi[6:12, :]
        nu_r2 = nu_r[3:6, :]
        f_B = evalf(f_B)
        tf_B2I = self.auv.compute_transformation_matrix(eta)
        R_B2I = evalf(tf_B2I[0:3, 0:3])
        
        ctrl_obj["S_kk"][:, :, idx] = evalf(skew(nu_r2))
        ctrl_obj["F_kk"][:, :, idx] = np.eye(3) - ctrl_obj["S_kk"][:, :, idx] * self.dt
        ctrl_obj["H_kk"][:, :, idx] = np.vstack((-ctrl_obj["S_kk"][:, :, idx], np.array([[0, 0, 1]]) @ R_B2I))
        
        if idx < self.window_size - 1: N = idx
        else: N = self.window_size - 1
        
        uco_idx = [i for i in range(idx - N + 1, idx + 1)]
        uco_idx.reverse()
        
        Phi = np.eye(3)
        for kk in uco_idx:
            Phi = Phi @ ctrl_obj["F_kk"][:, :, kk]
        Phi_inv = np.linalg.inv(Phi)
        
        if idx == 0:
            ctrl_obj["G"] = Phi_inv.T @ ctrl_obj["G0"] @ Phi_inv
        else:
            ctrl_obj["G"] = Phi_inv.T @ ctrl_obj["G_kk"][:, :, idx - 1] @ Phi_inv
            
        lb = -0.01
        ub = 0.01
        bnds = tuple([(lb, ub) for _ in range(self.thrusters)])
        
        sol = minimize(self.cost_fn, np.random.uniform(lb, ub, size=(self.thrusters,1)), args=(chi, f_B, ctrl_obj), bounds=bnds, method="SLSQP")
        # sol = minimize(self.cost_fn, np.zeros((self.thrusters,1)), args=(chi, f_B, ctrl_obj), bounds=bnds, method="SLSQP")
        u_next = sol.x
        
        chi_dot = self.auv.compute_nonlinear_dynamics(chi, u_next, f_B=f_B, f_est=self.est_flag, complete_model=self.model_type)
        chi_next = chi + chi_dot[0:12] * self.dt
        
        S_kk_next = evalf(skew(chi_next[9:12, :]))
        tf_B2I_est = self.auv.compute_transformation_matrix(chi_next[0:6, :])
        R_B2I_est = evalf(tf_B2I_est[0:3, 0:3])
        H_kk_next = np.vstack((-S_kk_next, np.array([[0, 0, 1]]) @ R_B2I_est))
        
        ctrl_obj["G_kk"][: ,:, idx] = H_kk_next.T @ np.linalg.inv(self.R_f_meas) @ H_kk_next + ctrl_obj["G"]
        
        return u_next, ctrl_obj
    
    def run_fse_mpc(self, idx, chi, x_ref, f_B, nu_w, nu_w_dot, ctrl_obj, kf):
        self.kf = kf
        return self.optimize(idx, chi, x_ref, f_B, nu_w, nu_w_dot, ctrl_obj)
    
    def optimize(self, idx, chi, x_ref, f_B, nu_w, nu_w_dot, ctrl_obj):
        eta = chi[0:6, :]
        # nu_r = chi[6:12, :]
        # nu_r2 = nu_r[3:6, :]
        # f_B = evalf(f_B)
        tf_B2I = self.auv.compute_transformation_matrix(eta)
        R_B2I = evalf(tf_B2I[0:3, 0:3])
        
        # xr = x_ref[0:6, :]
        # x0 = chi[0:6, :]
        xr = x_ref[0:12, :]
        x0 = chi[0:12, :]
        cost = 0
        
        opt = Opti()

        X = opt.variable(18, self.horizon+1)
        U = opt.variable(self.thrusters, self.horizon+1)
        # X0 = opt.parameter(6, 1)
        X0 = opt.parameter(12, 1)
        # ap_control_seq = np.zeros((8, self.horizon))

        # Only do horizon rolling if our horizon is greater than 1
        if (self.horizon > 1) and (self.previous_control is not None):
            # Shift all commands one over, since we executed the first control action
            initial_guess_control = np.roll(self.previous_control, -1, axis=1)
            initial_guess_state = np.roll(self.previous_state, -1, axis=1)

            # Set the final column to the same as the second to last column
            initial_guess_control[:,-1] = initial_guess_control[:,-2]
            initial_guess_state[:,-1] = initial_guess_state[:,-2]

            opt.set_initial(U, initial_guess_control)
            opt.set_initial(X, initial_guess_state)
            
        u, ctrl_obj = self.ap_control_cg(idx, chi, f_B, ctrl_obj)
        # chi_f_pred, _ = self.kf.predict(chi)
        chi_f_pred = evalf(f_B)
        chi_f_I = inv(R_B2I) @ chi_f_pred
                
        for k in range(self.horizon):
            chi_f_pred = self.get_flow(chi_f_I, X[0:6, k])
            
            # u, ctrl_obj = self.ap_control_cg(idx, X[:, k], chi_f_pred, ctrl_obj)
            # ap_control_seq[:, k] = u
            # chi_f_pred, _ = self.kf.predict(X[:, k])
            
            # U[:, k] = U[:, k]
            U[:, k] = U[:, k] + u
                
            # cost += (X[0:6, k] - xr).T @ self.P @ (X[0:6, k] - xr) 
            cost += (X[0:12, k] - xr).T @ self.P @ (X[0:12, k] - xr) 
            # cost += (U[:, k]).T @ self.Q @ (U[:, k])
            # cost += (U[:, k+1] - U[:, k]).T @ self.R @ (U[:, k+1] - U[:, k])
            
            # opt.subject_to(X[:, k+1] == chi_next)
            opt.subject_to(X[:, k+1] == self.forward_dynamics(X[:, k], U[:, k], evalf(f_B), nu_w, nu_w_dot, self.est_flag, self.model_type))
            # opt.subject_to(X[:, k+1] == self.forward_dynamics(X[:, k], U[:, k], chi_f_pred, self.est_flag, self.model_type))
            
            opt.subject_to(opt.bounded(self.xmin, X[0:12, k], self.xmax))
            opt.subject_to(opt.bounded(self.umin, U[:, k], self.umax))
            opt.subject_to(opt.bounded(self.dumin, (U[:, k+1] - U[:, k]), self.dumax))
            if (self.previous_control is not None) and (k == 0):
                opt.subject_to(opt.bounded(self.dumin, (U[:, k] - self.previous_control[:, k]), self.dumax))

        cost += (X[0:12, -1] - xr).T @ self.P @ (X[0:12, -1] - xr)
        
        opt.subject_to(opt.bounded(self.xmin, X[0:12, -1], self.xmax)) 
        # opt.subject_to(X[0:6, 0] == X0)
        opt.subject_to(X[0:12, 0] == X0)
        
        opt.set_value(X0, x0)
        opt.minimize(cost)
        
        options = {"ipopt" : {}}
        
        if self.log_quiet:
            options["ipopt"]["print_level"] = 0 
            options["print_time"] = 0
        else:
            options['show_eval_warnings'] = True

        opt.solver('ipopt', options)
        sol = opt.solve()
        
        u_next = sol.value(U)[:,0:1]
        # x_next = sol.value(X)[:,1:2]

        self.previous_control = sol.value(U)
        self.previous_state = sol.value(X)
        inst_cost = sol.value(cost)
        # thrust_force = evalf(mtimes(self.auv.tam, u_next)).full()

        # u_next= u
        # inst_cost = 0.0

        chi_dot = self.auv.compute_nonlinear_dynamics(chi, u_next, f_B, f_est=self.est_flag, complete_model=self.model_type)
        chi_next = chi + chi_dot[0:12] * self.dt
        
        S_kk_next = evalf(skew(chi_next[9:12, :]))
        tf_B2I_est = self.auv.compute_transformation_matrix(chi_next[0:6, :])
        R_B2I_est = evalf(tf_B2I_est[0:3, 0:3])
        H_kk_next = np.vstack((-S_kk_next, np.array([[0, 0, 1]]) @ R_B2I_est))
        
        ctrl_obj["G_kk"][: ,:, idx] = H_kk_next.T @ np.linalg.inv(self.R_f_meas) @ H_kk_next + ctrl_obj["G"]
                
        # return u_next, ctrl_obj, inst_cost, thrust_force
        return u_next, ctrl_obj, inst_cost
        