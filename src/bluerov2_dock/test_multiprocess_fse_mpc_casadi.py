import os
import pickle
import time
from copy import deepcopy
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import yaml
from casadi import SX, evalf, inv, mtimes, skew, vertcat
from filterpy.stats import plot_3d_covariance, plot_covariance
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from auv_hinsdale import AUV
from extended_kalman_filter import ExtendedKalmanFilter
from env_casadi import EnvironmentManager
from fse_mpc_casadi import FEMPC
from kalman import KalmanFilter
from particle_filter import ParticleFilter


class FEMPControl():
    def __init__(self):
        cwd = os.path.dirname(__file__)
        self.auv_yaml = cwd + "/../../config/auv_bluerov2_heavy.yaml"
        self.fse_mpc_yaml = cwd + "/../../config/mpc_bluerov2_heavy.yaml"
        self.test_params_yaml = cwd + "/../../config/test_params.yaml"

        self.test_params = self.load_params(self.test_params_yaml)
        self.num_trials = self.test_params["num_trials"]
        self.envs = self.test_params["environments"]
        self.algorithm = str(self.test_params["algorithm"])
        # self.initialize_params()

    def load_params(self, filename):
        f = open(filename, "r")
        test_params = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return test_params

    def initialize_params(self):
        # Change these values as desired
        self.x_ini = np.array([self.test_params["x0"]]).T
        self.x_ref = np.array([self.test_params["xr"]]).T
        self.wec_z_ref = self.x_ref[2][0]

        self.tolerance = self.test_params["tolerance"]
        self.path_length = 0.0
        self.p_times = [0]

        env_filename = "data/WECM/OceanCurrCase1.csv"
        self.read_env_data(env_filename)
        self.update_env(0)

        self.auv = AUV.load_params(self.auv_yaml)
        self.fse_mpc = FEMPC.load_params(self.auv_yaml, self.fse_mpc_yaml)
        self.env_man = EnvironmentManager(self.auv)
        self.env = self.env_man.init_env(self.env_name, self.env_type, self.env_val)
        self.env = self.env_man.handle_env(self.env, 0.0)
        self.env_flow_ini = self.env_man.get_env(self.x_ini[0:6, :], self.x_ini[6:12, :], self.env, 0)
        self.f_B_ini = vertcat(self.env_flow_ini['f_B'], SX.zeros((3, 1)))
        self.f1_ini = self.f_B_ini[0:3, :]
        self.x_ini[6:9, :] = -evalf(self.f1_ini)
        self.pf_ini = np.vstack((self.x_ini, evalf(self.f_B_ini)))
        self.ekf_ini = self.x_ini

        self.dt = self.fse_mpc.dt
        self.t_f = self.test_params["t_len"]
        self.t_span = np.arange(0.0, self.t_f, self.dt)

        self.static_target = self.test_params["static_target"]
        self.wec_amp = self.test_params["wec"]["A"]
        self.wec_omega = self.test_params["wec"]["omega"]

        n_f_state = self.f1_ini.shape[0]
        n_f_meas = len(self.fse_mpc.R_f_meas)
        n_auv_meas = len(self.fse_mpc.R_auv_meas)

        n_particles = self.test_params["pf"]["particles"]
        pf_init_cov = float(self.test_params["pf"]["cov"])

        self.kf = KalmanFilter(evalf(self.f1_ini), self.fse_mpc.Q_f, self.fse_mpc.R_f_meas,
                               self.f_state_trans_model, self.f_meas_model)

        self.ekf = ExtendedKalmanFilter(evalf(self.ekf_ini), self.fse_mpc.Q_auv_state, self.fse_mpc.R_auv_meas,
                                        self.auv_state_model_ekf, self.auv_meas_model_ekf)

        self.pf = ParticleFilter(self.fse_mpc.Q_auv_state, self.fse_mpc.R_auv_meas,
                                 self.auv_state_model, self.auv_meas_model)
        self.pf.initialize(n_particles, self.pf_ini, pf_init_cov * np.eye(self.pf_ini.shape[0]))

        self.ctrl_obj = {}
        self.ctrl_obj["F_kk"] = np.zeros((3, 3, len(self.t_span)))
        self.ctrl_obj["S_kk"] = np.zeros((3, 3, len(self.t_span)))
        self.ctrl_obj["H_kk"] = np.zeros((n_f_meas, n_f_state, len(self.t_span)))
        self.ctrl_obj["G_kk"] = np.zeros((3, 3, len(self.t_span)))
        self.ctrl_obj["G0"] = np.linalg.inv(np.eye(n_f_state))

        self.wec_data = {"state": {}}
        self.wec_data["state"]["eta"] = np.zeros((6, len(self.t_span) + 1))
        self.wec_data["state"]["nu_r"] = np.zeros((6, len(self.t_span) + 1))

        self.nav_data = {"state": {},
                         "control": {},
                         "analysis": {}}
        self.nav_data["state"]["t"] = self.t_span
        self.nav_data["state"]["eta"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["nu"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["nu_r"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["f_B"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["f_I"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["nu_I"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["control"]["u"] = np.zeros((self.fse_mpc.thrusters, len(self.t_span)))
        self.nav_data["control"]["u1"] = np.zeros((self.fse_mpc.thrusters, len(self.t_span)))
        self.nav_data["control"]["u2"] = np.zeros((self.fse_mpc.thrusters, len(self.t_span)))
        self.nav_data["analysis"]["eta_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["nu_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["nu_r_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["f_B_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["f_I_dot"] = np.zeros((3, len(self.t_span)))
        self.nav_data["analysis"]["nu_I_dot"] = np.zeros((3, len(self.t_span)))
        self.nav_data["analysis"]["inst_cost"] = np.zeros((1, len(self.t_span)))
        self.nav_data["analysis"]["thrust_force"] = np.zeros((6, len(self.t_span)))

        self.nav_est_data = {"state": {},
                             "meas": {},
                             "analysis": {}}
        self.nav_est_data["state"]["t"] = self.t_span[0:-1]
        self.nav_est_data["state"]["eta"] = np.zeros((6, len(self.t_span)))
        self.nav_est_data["state"]["nu_r"] = np.zeros((6, len(self.t_span)))
        self.nav_est_data["state"]["f_B"] = np.zeros((3, len(self.t_span)))
        self.nav_est_data["state"]["f_I"] = np.zeros((3, len(self.t_span)))
        self.nav_est_data["state"]["t_pred"] = self.t_span
        self.nav_est_data["state"]["eta_pred"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_est_data["state"]["nu_r_pred"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_est_data["state"]["f_B_pred"] = np.zeros((3, len(self.t_span) + 1))
        self.nav_est_data["state"]["f_I_pred"] = np.zeros((3, len(self.t_span) + 1))
        self.nav_est_data["meas"]["zeta_f"] = np.zeros((n_f_meas, len(self.t_span)))
        self.nav_est_data["meas"]["zeta_auv"] = np.zeros((n_auv_meas, len(self.t_span)))
        self.nav_est_data["analysis"]["eta_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_est_data["analysis"]["nu_r_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_est_data["analysis"]["f_B_dot"] = np.zeros((3, len(self.t_span)))
        self.nav_est_data["analysis"]["f_I_dot"] = np.zeros((3, len(self.t_span)))

        self.env_data = {"state": {},
                         "analysis": {}}
        self.env_data["state"]["t"] = self.t_span[0:-1]
        self.env_data["state"]["f_I"] = np.zeros((3, len(self.t_span)))
        self.env_data["state"]["f_B"] = np.zeros((3, len(self.t_span)))
        self.env_data["state"]["nu_w"] = np.zeros((3, len(self.t_span)))
        self.env_data["analysis"]["f_I_dot"] = np.zeros((3, len(self.t_span)))
        self.env_data["analysis"]["f_B_dot"] = np.zeros((3, len(self.t_span)))
        self.env_data["analysis"]["nu_w_dot"] = np.zeros((3, len(self.t_span)))

        self.env_est_data = {"state": {},
                             "analysis": {}}
        self.env_est_data["state"]["t"] = self.t_span[0:-1]
        self.env_est_data["state"]["f_I"] = np.zeros((3, len(self.t_span)))
        self.env_est_data["state"]["f_B"] = np.zeros((3, len(self.t_span)))
        self.env_est_data["analysis"]["f_I_dot"] = np.zeros((3, len(self.t_span)))
        self.env_est_data["analysis"]["f_B_dot"] = np.zeros((3, len(self.t_span)))

        self.est_data = {"analysis": {}}
        self.est_data["P_f_pred"] = np.zeros((3, 3, len(self.t_span) + 1))
        self.est_data["P_f_est"] = np.zeros((3, 3, len(self.t_span) + 1))
        self.est_data["P_auv_pred"] = np.zeros((12, 12, len(self.t_span) + 1))
        self.est_data["P_auv_est"] = np.zeros((12, 12, len(self.t_span) + 1))
        self.est_data["analysis"]["est_eta_diff"] = np.zeros((6, len(self.t_span)))
        self.est_data["analysis"]["est_eta_sqdiff"] = np.zeros((1, len(self.t_span)))
        self.est_data["analysis"]["est_nu_r_diff"] = np.zeros((6, len(self.t_span)))
        self.est_data["analysis"]["est_nu_r_sqdiff"] = np.zeros((1, len(self.t_span)))
        self.est_data["analysis"]["est_f_B_diff"] = np.zeros((3, len(self.t_span)))
        self.est_data["analysis"]["est_f_B_dot_diff"] = np.zeros((3, len(self.t_span)))
        self.est_data["analysis"]["est_f_B_sqdiff"] = np.zeros((1, len(self.t_span)))
        self.est_data["analysis"]["est_f_B_dot_sqdiff"] = np.zeros((1, len(self.t_span)))

        self.opt_data = {}

    def read_env_data(self, filename):
        env_data = np.genfromtxt(filename, delimiter=",", dtype=np.float64)
        self.wave_vel_data = env_data.T

    def update_env(self, i):
        if i > 0:
            self.wave_acc = (self.wave_vel_data[:, i].reshape(-1, 1) - self.wave_vel) / self.dt

        self.wave_vel = self.wave_vel_data[:, i].reshape(-1, 1)

    def update_wec_state(self, i):
        env_next = self.env_man.handle_env(self.env, self.t_span[i])
        env_flow = self.env_man.get_env(self.x_ref[0:6, :], self.x_ref[6:12, :], env_next, self.t_span[i])

        if self.static_target:
            v_ref = np.array([[0., 0., 0., 0., 0., 0.]]).T
        else:
            next_wec_z = self.wec_amp * np.sin(self.wec_omega * self.t_span[i]) + self.wec_z_ref
            next_wec_psi = self.wec_omega * self.wec_amp * np.cos(self.wec_omega * self.t_span[i])
            self.x_ref[2][0] = next_wec_z
            v_ref = np.array([[0., 0., next_wec_psi, 0., 0., 0.]]).T

        tf_wec_RB2I = self.auv.compute_transformation_matrix(self.x_ref[0:6, :])
        flow_bf_wec = evalf(mtimes(inv(tf_wec_RB2I), vertcat(env_flow['f'], SX.zeros(3, 1))))
        self.x_ref[6:12, :] = v_ref - flow_bf_wec

    def f_state_trans_model(self, x):
        nu_r2 = x[9:12, :]
        S = skew(nu_r2)
        A = SX.eye(3) - S * self.dt
        A = np.array(evalf(A))
        return A

    def f_meas_model(self, x):
        eta = x[0:6, :]
        nu_r2 = x[9:12, :]
        S = skew(nu_r2)
        tf_B2I = self.auv.compute_transformation_matrix(eta)
        R_B2I = tf_B2I[0:3, 0:3]
        H = vertcat(-S, SX([[0, 0, 1]]) @ R_B2I)
        H = np.array(evalf(H))
        return H

    def auv_state_model(self, chi, u, f_B):
        chi = chi.reshape(-1, 1)
        chi_dot = evalf(self.auv.compute_nonlinear_dynamics(chi, u, f_B, f_est=True, complete_model=True))
        chi_next = (chi + chi_dot * self.dt).full()
        chi_next[3:6, :] = self.wrap_pi2negpi(chi_next[3:6, :])
        chi_next[6:12, :] = np.clip(chi_next[6:12, :], self.fse_mpc.xmin[6:12], self.fse_mpc.xmax[6:12]).reshape(-1, 1)
        return chi_next

    def auv_meas_model(self, chi, u, f_B):
        chi = chi.reshape(-1, 1)
        eta = chi[0:6, :]
        nu_r = chi[6:12, :]
        nu = chi[6:12, :] + chi[12:18, :]
        chi_dot = evalf(self.auv.compute_nonlinear_dynamics(chi, u, f_B, f_est=True, complete_model=True)).full()
        chi_dot[6:12, :] = np.clip(chi_dot[6:12, :], self.fse_mpc.xmin[6:12], self.fse_mpc.xmax[6:12]).reshape(-1, 1)
        f1_B_dot = chi_dot[12:15, :]
        nu_r1_dot = chi_dot[6:9, :]
        acc = f1_B_dot + nu_r1_dot

        zeta = np.zeros((15, 1))
        zeta[0:2, 0] = eta[0:2, 0]
        zeta[2, 0] = eta[2, 0]
        zeta[3:6, 0] = eta[3:6, 0]
        zeta[6:9, 0] = nu_r[0:3, 0]
        zeta[9:12, 0] = nu[3:6, 0]
        zeta[12:15, 0] = evalf(acc).full().flatten()
        return zeta

    def auv_state_model_ekf(self, chi, u, f_B, nu_w, nu_w_dot):
        # chi = chi.reshape(-1, 1)
        chi_dot = self.auv.compute_nonlinear_dynamics(
            chi, u, f_B, nu_w=nu_w, nu_w_dot=nu_w_dot, f_est=True, complete_model=True)
        chi_next = chi + (chi_dot * self.dt)[0:12, 0]
        # chi_next[3:6, :] = self.wrap_pi2negpi(chi_next[3:6, :])
        # chi_next[6:12, :] = np.clip(chi_next[6:12, :], self.fse_mpc.xmin[6:12], self.fse_mpc.xmax[6:12]).reshape(-1,1)
        return chi_next

    def auv_meas_model_ekf(self, chi, u, f_B, nu_w, nu_w_dot):
        # chi = chi.reshape(-1, 1)
        eta = chi[0:6, :]
        nu_r = chi[6:12, :]
        # skew_mtx = SX.eye(3)
        # skew_mtx = -skew(nu_r[3:6])
        nu_2 = nu_r[3:6]  # + mtimes(skew_mtx, f_B)
        chi_dot = self.auv.compute_nonlinear_dynamics(
            chi, u, f_B, nu_w=nu_w, nu_w_dot=nu_w_dot, f_est=True, complete_model=True)
        # chi_dot[6:12, :] = np.clip(chi_dot[6:12, :], self.fse_mpc.xmin[6:12], self.fse_mpc.xmax[6:12]).reshape(-1,1)
        f1_B_dot = chi_dot[12:15, :]
        nu_r1_dot = chi_dot[6:9, :]
        acc = f1_B_dot + nu_r1_dot

        zeta = SX.sym('zeta', 10, 1)
        # zeta[0, 0] = eta[2, 0]
        # zeta[1:4, 0] = eta[3:6, 0]
        # zeta[4:7, 0] = nu_r[3:6, 0]
        # zeta[7:10, 0] = acc
        # zeta[0:2, 0] = eta[0:2, 0]
        zeta[0, 0] = eta[2, 0]
        zeta[1:4, 0] = eta[3:6, 0]
        zeta[4:7, 0] = nu_2
        # zeta[9:12, 0] = nu[3:6, 0]
        zeta[7:10, 0] = acc
        return zeta

    def flow_model_est(self, chi, chi_f):
        nu_r = chi[6:12, :]
        f1_B = chi_f
        S = skew(nu_r[3:6, :])
        f1_B_dot = -mtimes(S, f1_B)
        f1_B_dot = np.array(evalf(f1_B_dot))
        return f1_B_dot

    def wrap_pi2negpi(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def run_fse_mpc(self, e):
        self.env_val = e[0]
        self.env_name = e[1]
        self.env_type = e[2]

        for t_id in range(self.num_trials):
            self.initialize_params()

            x_true = self.x_ini
            f_B_true = self.f_B_ini
            chi_true = vertcat(x_true, f_B_true)
            chi_true = np.array(evalf(chi_true))

            self.fse_mpc.reset()

            self.wec_data["state"]["eta"][:, 0] = self.x_ref[0:6, :].flatten()
            self.wec_data["state"]["nu_r"][:, 0] = self.x_ref[6:12, :].flatten()
            self.nav_data["state"]["eta"][:, 0] = chi_true[0:6, :].flatten()
            self.nav_data["state"]["nu_r"][:, 0] = chi_true[6:12, :].flatten()
            self.nav_data["state"]["f_B"][:, 0] = chi_true[12:18, :].flatten()
            self.nav_data["state"]["nu"][:, 0] = (chi_true[6:12, :] + chi_true[12:18, :]).flatten()
            self.nav_est_data["state"]["eta_pred"][:, 0] = self.pf.X[0:6].flatten()
            self.nav_est_data["state"]["nu_r_pred"][:, 0] = self.pf.X[6:12].flatten()
            self.nav_est_data["state"]["f_B_pred"][:, 0] = self.kf.x.full().flatten()
            self.est_data["P_f_pred"][:, :, 0] = self.kf.P
            self.est_data["P_f_est"][:, :, 0] = self.kf.P
            self.est_data["P_auv_pred"][:, :, 0] = self.ekf.P
            self.est_data["P_auv_est"][:, :, 0] = self.ekf.P
            # self.est_data["P_auv_pred"][:, :, 0] = self.pf.P
            # self.est_data["P_auv_est"][:, :, 0] = self.pf.P

            i = 0
            process_t0 = time.perf_counter()

            for i in range(len(self.t_span)):
                # while self.distance > self.tolerance:
                self.update_wec_state(i)
                self.update_env(i + 1)

                # self.distance = np.linalg.norm(self.x_ini - self.x_ref)

                eta_pred = self.nav_est_data["state"]["eta_pred"][:, i]
                nu_r_pred = self.nav_est_data["state"]["nu_r_pred"][:, i]
                f_B_pred = self.nav_est_data["state"]["f_B_pred"][:, i]
                P_auv_pred = evalf(self.est_data["P_auv_pred"][:, :, i])
                chi_auv_pred = vertcat(eta_pred, nu_r_pred)

                self.env = self.env_man.handle_env(self.env, self.t_span[i])
                self.env_flow = self.env_man.get_env(chi_true[0:6, :], chi_true[6:12, :], self.env, self.t_span[i])

                # u, ctrl_obj, inst_cost = self.fse_mpc.run_fse_mpc(i, chi_auv_pred, self.x_ref, f_B_pred, self.ctrl_obj, self.kf)
                # u, ctrl_obj, inst_cost = self.fse_mpc.run_fse_mpc(i, chi_auv_pred, self.x_ref, self.env_flow["f_B"], self.ctrl_obj, self.kf)
                # u, ctrl_obj, inst_cost = self.fse_mpc.run_fse_mpc(i, chi_true, self.x_ref, self.env_flow["f_B"], self.ctrl_obj, self.kf)
                u, ctrl_obj, inst_cost = self.fse_mpc.run_fse_mpc(
                    i, chi_auv_pred, self.x_ref, f_B_pred, self.wave_vel, self.wave_acc, self.ctrl_obj, self.kf)
                self.ctrl_obj = ctrl_obj

                if self.fse_mpc.dt_int == "euler":
                    # chi_dot = self.auv.compute_nonlinear_dynamics(chi_true, u, self.env_flow["f_B"],
                    #                                             self.env_flow["f_B_dot"],
                    #                                             f_est=False, complete_model=True)
                    chi_dot = self.auv.compute_nonlinear_dynamics(chi_true, u, self.env_flow["f_B"],
                                                                  self.env_flow["f_B_dot"], self.wave_vel, self.wave_acc,
                                                                  f_est=False, complete_model=True)
                elif self.fse_mpc.dt_int == "rk4":
                    pass

                chi_dot = np.array(evalf(chi_dot))

                chi_true += chi_dot * self.dt
                chi_true[3:6, :] = self.wrap_pi2negpi(chi_true[3:6, :])
                chi_true[6:12, :] = np.clip(chi_true[6:12, :], self.fse_mpc.xmin[6:12],
                                            self.fse_mpc.xmax[6:12]).reshape(-1, 1)

                self.nav_data["state"]["eta"][:, i + 1] = chi_true[0:6, :].flatten()
                self.nav_data["state"]["nu_r"][:, i + 1] = chi_true[6:12, :].flatten()
                self.nav_data["state"]["f_B"][:, i + 1] = chi_true[12:18, :].flatten()
                self.nav_data["state"]["nu"][:, i + 1] = (chi_true[6:12, :] + chi_true[12:18, :]).flatten()
                self.nav_data["control"]["u"][:, i] = u.flatten()
                self.nav_data["analysis"]["eta_dot"][:, i] = chi_dot[0:6, :].flatten()
                self.nav_data["analysis"]["nu_r_dot"][:, i] = chi_dot[6:12, :].flatten()
                self.nav_data["analysis"]["f_B_dot"][:, i] = chi_dot[12:18, :].flatten()
                self.nav_data["analysis"]["nu_dot"][:, i] = (chi_dot[6:12, :] + chi_dot[12:18, :]).flatten()
                self.nav_data["analysis"]["inst_cost"][:, i] = inst_cost
                # self.nav_data["analysis"]["thrust_force"][:, i] = thrust_force.flatten()

                self.env_data["state"]["nu_w"][:, i] = self.wave_vel.flatten()
                self.env_data["analysis"]["nu_w_dot"][:, i] = self.wave_acc.flatten()
                self.env_data["state"]["f_B"][:, i] = evalf(self.env_flow['f_B']).full().flatten()
                self.env_data["analysis"]["f_B_dot"][:, i] = evalf(self.env_flow['f_B_dot']).full().flatten()

                eta_true = self.nav_data["state"]["eta"][:, i].reshape(-1, 1)
                nu_r_true = self.nav_data["state"]["nu_r"][:, i].reshape(-1, 1)
                x_true = np.vstack((eta_true, nu_r_true))
                nu_true = self.nav_data["state"]["nu"][:, i]
                f_B_true = self.env_data["state"]["f_B"][:, i]
                eta_dot_true = self.nav_data["analysis"]["eta_dot"][:, i].reshape(-1, 1)
                nu_r_dot_true = self.nav_data["analysis"]["nu_r_dot"][:, i].reshape(-1, 1)
                nu_dot_true = self.nav_data["analysis"]["nu_dot"][:, i]
                f_B_dot_true = self.env_data["analysis"]["f_B_dot"][:, i]
                chi_auv_true = np.vstack((eta_true, nu_r_true))
                chi_auv_dot_true = np.vstack((eta_dot_true, nu_r_dot_true))
                chi_f_true = f_B_true
                chi_f_dot_true = f_B_dot_true

                chi_pred_dot = self.auv.compute_nonlinear_dynamics(chi_auv_pred, u, f_B_true, nu_w=self.wave_vel, nu_w_dot=self.wave_acc,
                                                                   f_est=True, complete_model=True)
                # chi_pred_dot = self.auv.compute_nonlinear_dynamics(chi_auv_pred, u, f_B_pred,
                #                                                     f_est=True, complete_model=True)
                chi_auv_pred_dot = evalf(chi_pred_dot)[0:12, :].full()
                chi_auv_pred_dot[6:12, :] = np.clip(chi_auv_pred_dot[6:12, :],
                                                    self.fse_mpc.xmin[6:12], self.fse_mpc.xmax[6:12]).reshape(-1, 1)

                nu_r_pred_dot = chi_auv_pred_dot[6:12, :]
                nu_r_pred = chi_auv_pred[6:12]

                tf_B2I_est = self.auv.compute_transformation_matrix(chi_auv_pred[0:6, :])
                R_B2I_est = evalf(tf_B2I_est[0:3, 0:3])

                eta_z_meas = eta_true[2] + np.sqrt(self.fse_mpc.R_z) * np.random.randn()
                eta_z_dot_meas = eta_dot_true[2] + np.sqrt(self.fse_mpc.R_dr) * np.random.randn()
                ang_vel_meas = nu_true[3:6] + np.diag(np.sqrt(self.fse_mpc.R_angvel)) * np.random.randn()
                eta_att_meas = eta_true[3:6, 0] + np.diag(np.sqrt(self.fse_mpc.R_att)) * np.random.randn()
                lin_acc_meas = nu_dot_true[0:3] + np.diag(np.sqrt(self.fse_mpc.R_linacc)) * np.random.randn()
                xy_meas = eta_true[0:2, 0] + np.diag(np.sqrt(self.fse_mpc.R_xy)) * np.random.randn()
                lin_vel_meas = nu_r_true[0:3, 0] + np.diag(np.sqrt(self.fse_mpc.R_linvel)) * np.random.randn()

                xy_meas = xy_meas.reshape(2, 1)
                lin_vel_meas = lin_vel_meas.reshape(3, 1)
                ang_vel_meas = ang_vel_meas.reshape(3, 1)
                eta_att_meas = eta_att_meas.reshape(3, 1)
                lin_acc_meas = lin_acc_meas.reshape(3, 1)

                zeta_auv = np.vstack((eta_z_meas, eta_att_meas, ang_vel_meas, lin_acc_meas))
                # zeta_auv = np.vstack((xy_meas, eta_z_meas, eta_att_meas, lin_vel_meas, lin_acc_meas))

                zeta_f = np.vstack((lin_acc_meas - nu_r_pred_dot[0:3, :],
                                    eta_z_dot_meas - np.array([[0, 0, 1]]) @ (R_B2I_est @ nu_r_pred[0:3, :])))

                chi_f_est, P_f_est = self.kf.update(chi_auv_pred, zeta_f, P_auv_pred)

                chi_auv_est, P_auv_est = self.ekf.update(
                    zeta_auv, chi_auv_pred, u, evalf(
                        self.env_flow['f_B']).full(), self.wave_vel, self.wave_acc)
                # chi_auv_est, P_auv_est = self.ekf.update(zeta_auv, chi_auv_pred, u, chi_f_est)
                chi_auv_est = chi_auv_est.full()

                # chi_auv_est, P_auv_est = self.pf.update(zeta_auv, u, evalf(self.env_flow['f_B']).full())
                # chi_auv_est, P_auv_est = self.pf.update(zeta_auv, u, chi_f_est)
                chi_auv_est[6:12] = np.clip(chi_auv_est[6:12], self.fse_mpc.xmin[6:12], self.fse_mpc.xmax[6:12])
                chi_auv_est = chi_auv_est.reshape(-1, 1)
                chi_auv_est[3:6, :] = self.wrap_pi2negpi(chi_auv_est[3:6, :])

                # chi_est_dot = self.auv.compute_nonlinear_dynamics(chi_auv_est, u,
                #                                                 evalf(self.env_flow['f_B']).full(),
                #                                                 evalf(self.env_flow['f_B_dot']).full(),
                #                                                 complete_model=True)
                chi_est_dot = self.auv.compute_nonlinear_dynamics(chi_auv_est, u, chi_f_est, nu_w=self.wave_vel, nu_w_dot=self.wave_acc,
                                                                  f_est=True, complete_model=True)
                chi_auv_est_dot = evalf(chi_est_dot)[0:12, :]
                # chi_f_est_dot = self.flow_model_est(chi_auv_true, chi_f_est)
                chi_f_est_dot = self.flow_model_est(chi_auv_est, chi_f_est)
                # chi_f_est_dot = self.flow_model_est(chi_auv_est, f_B_true)

                # chi_auv_pred, P_auv_pred = self.pf.predict(u, evalf(self.env_flow['f_B']).full())
                # chi_auv_pred, P_auv_pred = self.pf.predict(u, chi_f_est)
                chi_auv_pred, P_auv_pred = self.ekf.predict(chi_auv_est, u, evalf(
                    self.env_flow['f_B']).full(), self.wave_vel, self.wave_acc)
                # chi_auv_pred, P_auv_pred = self.ekf.predict(chi_auv_est, u, chi_f_est)
                chi_auv_pred = chi_auv_pred.full()
                chi_auv_pred[6:12, :] = np.clip(chi_auv_pred[6:12, :], self.fse_mpc.xmin[6:12],
                                                self.fse_mpc.xmax[6:12]).reshape(-1, 1)
                chi_auv_pred[3:6, :] = self.wrap_pi2negpi(chi_auv_pred[3:6, :])

                chi_f_pred, P_f_pred = self.kf.predict(chi_auv_est)

                self.nav_est_data["meas"]["zeta_auv"][:, i] = zeta_auv.flatten()
                self.nav_est_data["meas"]["zeta_f"][:, i] = zeta_f.flatten()
                self.nav_est_data["state"]["eta"][:, i] = chi_auv_est[0:6].flatten()
                self.nav_est_data["state"]["nu_r"][:, i] = chi_auv_est[6:12].flatten()
                self.nav_est_data["state"]["f_B"][:, i] = chi_f_est.full().flatten()
                self.nav_est_data["analysis"]["eta_dot"][:, i] = chi_auv_est_dot[0:6].full().flatten()
                self.nav_est_data["analysis"]["nu_r_dot"][:, i] = chi_auv_est_dot[6:12].full().flatten()
                self.nav_est_data["analysis"]["f_B_dot"][:, i] = chi_f_est_dot.flatten()
                self.nav_est_data["state"]["eta_pred"][:, i + 1] = chi_auv_pred[0:6].flatten()
                self.nav_est_data["state"]["nu_r_pred"][:, i + 1] = chi_auv_pred[6:12].flatten()
                self.nav_est_data["state"]["f_B_pred"][:, i + 1] = chi_f_pred.full().flatten()

                self.env_est_data["analysis"]["f_B_dot"][:, i] = chi_f_est_dot.flatten()

                self.est_data["P_f_pred"][:, :, i + 1] = P_f_pred
                self.est_data["P_f_est"][:, :, i + 1] = P_f_est
                self.est_data["P_auv_pred"][:, :, i + 1] = evalf(P_auv_pred).full()
                self.est_data["P_auv_est"][:, :, i + 1] = evalf(P_auv_est).full()
                self.est_data["analysis"]["est_eta_diff"][:, i] = (chi_auv_true[0:6, :] - chi_auv_est[0:6, :]).flatten()
                self.est_data["analysis"]["est_eta_sqdiff"][:, i] = np.linalg.norm(
                    chi_auv_true[0:6, :] - chi_auv_est[0:6, :])
                self.est_data["analysis"]["est_nu_r_diff"][:, i] = (
                    chi_auv_true[6:12, :] - chi_auv_est[6:12, :]).flatten()
                self.est_data["analysis"]["est_nu_r_sqdiff"][:, i] = np.linalg.norm(
                    chi_auv_true[6:12, :] - chi_auv_est[6:12, :])
                # self.est_data["analysis"]["est_chi_dot_diff"][:, i] = (f_B_dot_true - chi_f_est_dot.flatten())
                # self.est_data["analysis"]["est_chi_dot_sqdiff"][:, i] = np.linalg.norm(f_B_dot_true - chi_f_est_dot.flatten())
                self.est_data["analysis"]["est_f_B_diff"][:, i] = (f_B_true - chi_f_est.full().flatten())
                self.est_data["analysis"]["est_f_B_sqdiff"][:,
                                                            i] = np.linalg.norm(f_B_true - chi_f_est.full().flatten())
                self.est_data["analysis"]["est_f_B_dot_diff"][:, i] = (f_B_dot_true - chi_f_est_dot.flatten())
                self.est_data["analysis"]["est_f_B_dot_sqdiff"][:,
                                                                i] = np.linalg.norm(f_B_dot_true - chi_f_est_dot.flatten())

                self.path_length += np.linalg.norm(chi_true[0:3, 0] - chi_auv_true[0:3, 0])
                self.distance = np.linalg.norm(chi_auv_est[0:6, :] - self.x_ref[0:6, :])
                # self.distance = np.linalg.norm(chi_true[0:12, :] - self.x_ref[0:12, :])

                self.wec_data["state"]["eta"][:, i] = self.x_ref[0:6, :].flatten()
                self.wec_data["state"]["nu_r"][:, i] = self.x_ref[6:12, :].flatten()

                comp_time = time.perf_counter() - process_t0
                self.p_times.append(comp_time)

                print(f"T = {round(self.t_span[i],3)}s, Time Index = {i}")
                print(f"Computation Time = {round(comp_time,3)}s")
                print("----------------------------------------------")
                print(f"FE MPC Contol Input: {np.round(u, 3).T}")
                print("----------------------------------------------")
                print(f"True Vehicle Pose: {np.round(chi_auv_true[0:6], 3).T}")
                print(f"Est Vehicle Pose: {np.round(chi_auv_est[0:6], 3).T}")
                print("Diff: ", np.round(self.est_data["analysis"]["est_eta_diff"][:, i], 6))
                print("Sq Diff: ", np.round(self.est_data["analysis"]["est_eta_sqdiff"][:, i], 6))
                print("----------------------------------------------")
                print(f"True Vehicle Velocity: {np.round(chi_auv_true[6:12], 3).T}")
                print(f"Est Vehicle Velocity: {np.round(chi_auv_est[6:12], 3).T}")
                print("Diff: ", np.round(self.est_data["analysis"]["est_nu_r_diff"][:, i], 6))
                print("Sq Diff: ", np.round(self.est_data["analysis"]["est_nu_r_sqdiff"][:, i], 6))
                print("----------------------------------------------")
                print("WEC Pose: ", self.wec_data["state"]["eta"][:, i].T)
                print("WEC Velocity: ", self.wec_data["state"]["nu_r"][:, i].T)
                print("----------------------------------------------")
                print(f"Path length: {np.round(self.path_length, 3)}")
                print(f"(WEC-AUV) Difference in state space: {np.round(self.distance, 3)}")
                print("----------------------------------------------")
                print(f"True Flow Velocity (f1): {f_B_true}")
                print(f"Estimated Flow Velocity (f1'): {chi_f_est.T}")
                print("Diff: ", self.est_data["analysis"]["est_f_B_diff"][:, i])
                print("Sq Diff: ", self.est_data["analysis"]["est_f_B_sqdiff"][:, i])
                print("----------------------------------------------")
                print(f"True Flow Acc (f1_dot): {f_B_dot_true}")
                print(f"Estimated Flow Acc (f1'_dot): {chi_f_est_dot.T}")
                print("Diff: ", self.est_data["analysis"]["est_f_B_dot_diff"][:, i])
                print("Sq Diff: ", self.est_data["analysis"]["est_f_B_dot_sqdiff"][:, i])
                print("----------------------------------------------")
                print("")

                i += 1

                if self.distance < self.tolerance:
                    break

            self.opt_data["comp_time"] = comp_time
            self.opt_data["path_length"] = self.path_length
            self.opt_data["opt_index"] = i
            self.opt_data["horizon"] = self.fse_mpc.horizon
            self.opt_data["dt"] = self.fse_mpc.dt
            self.opt_data["full_body"] = self.fse_mpc.model_type
            self.opt_data["f_est"] = self.fse_mpc.est_flag

            # self.save_logs(t_id)
            # self.plot_graphs()

        return True

    def save_logs(self, t_id):
        log_dir = "./data/" + self.algorithm + "/" + self.env_name + "/trial_" + str(t_id) + "/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(log_dir + '/opt_data.pkl', 'wb') as f:
            pickle.dump(self.opt_data, f)

        with open(log_dir + '/nav_data.pkl', 'wb') as f:
            pickle.dump(self.nav_data, f)

        with open(log_dir + '/est_data.pkl', 'wb') as f:
            pickle.dump(self.est_data, f)

        with open(log_dir + '/env_data.pkl', 'wb') as f:
            pickle.dump(self.env_data, f)

        with open(log_dir + '/wec_data.pkl', 'wb') as f:
            pickle.dump(self.wec_data, f)

        with open(log_dir + '/env_est_data.pkl', 'wb') as f:
            pickle.dump(self.env_est_data, f)

        with open(log_dir + '/nav_est_data.pkl', 'wb') as f:
            pickle.dump(self.nav_est_data, f)

    def plot_graphs(self):
        fig = plt.figure(dpi=100)
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_f_B_diff"][0, :], "c--", label="Error - u")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_f_B_diff"][1, :], "m--", label="Error - v")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_f_B_diff"][2, :], "k--", label="Error - w")
        plt.xlabel("Timestep [s]")
        plt.ylabel("Flow Estimation Error [m/s]")
        plt.legend()

        fig = plt.figure(dpi=100)
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]
                 ["est_f_B_dot_diff"][0, :], "c--", label="Error - u_dot")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]
                 ["est_f_B_dot_diff"][1, :], "m--", label="Error - v_dot")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]
                 ["est_f_B_dot_diff"][2, :], "k--", label="Error - w_dot")
        plt.xlabel("Timestep [s]")
        plt.ylabel("Flow Acc Estimation Error [m/s^2]")
        plt.legend()

        fig = plt.figure(dpi=100)
        plt.plot(self.nav_data["state"]["t"], np.sqrt(self.est_data["P_f_est"][0, 0, :-1]), "c--", label="u_std")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_f_B_diff"][0, :], "m--", label="u_est_err")
        plt.xlabel("Timestep [s]")
        plt.ylabel("Errors")
        plt.legend()

        fig = plt.figure(dpi=100)
        plt.plot(self.nav_data["state"]["t"], np.sqrt(self.est_data["P_f_est"][1, 1, :-1]), "c--", label="v_std")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_f_B_diff"][1, :], "m--", label="v_est_err")
        plt.xlabel("Timestep [s]")
        plt.ylabel("Errors")
        plt.legend()

        fig = plt.figure(dpi=100)
        plt.plot(self.nav_data["state"]["t"], np.sqrt(self.est_data["P_f_est"][2, 2, :-1]), "c--", label="w_std")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_f_B_diff"][2, :], "m--", label="w_est_err")
        plt.xlabel("Timestep [s]")
        plt.ylabel("Errors")
        plt.legend()

        fig = plt.figure(dpi=100)
        plt.plot(self.nav_data["state"]["t"], np.sqrt(self.est_data["P_auv_est"][0, 0, :-1]), "c--", label="x_std")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_eta_diff"][0, :], "m--", label="x_est_err")
        plt.xlabel("Timestep [s]")
        plt.ylabel("Errors")
        plt.legend()

        fig = plt.figure(dpi=100)
        plt.plot(self.nav_data["state"]["t"], np.sqrt(self.est_data["P_auv_est"][1, 1, :-1]), "c--", label="y_std")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_eta_diff"][1, :], "m--", label="y_est_err")
        plt.xlabel("Timestep [s]")
        plt.ylabel("Errors")
        plt.legend()

        fig = plt.figure(dpi=100)
        plt.plot(self.nav_data["state"]["t"], np.sqrt(self.est_data["P_auv_est"][2, 2, :-1]), "c--", label="z_std")
        plt.plot(self.nav_data["state"]["t"], self.est_data["analysis"]["est_eta_diff"][2, :], "m--", label="z_est_err")
        plt.xlabel("Timestep [s]")
        plt.ylabel("Errors")
        plt.legend()

        fig = plt.figure(dpi=100)
        for i in range(self.nav_est_data["state"]["f_B"].shape[1]):
            mean = self.nav_est_data["state"]["f_B"][0:2, i].reshape(2, 1)
            cov = self.est_data["P_f_est"][0:2, 0:2, i].reshape(2, 2)
            plot_covariance(mean=mean, cov=cov)
            plt.plot(self.env_data["state"]["f_B"][0, i], self.env_data["state"]["f_B"][1, i])

        plt.show()

    def main(self):
        test_cases = []
        for _, e in self.envs.items():
            env_val = np.array(e["current"]["value"], dtype=np.float64)
            env_name = str(e["current"]["name"])
            env_type = str(e["current"]["type"])
            case = [env_val, env_name, env_type]
            test_cases.append(case)

        successes = self.run_fse_mpc(test_cases[0])

        # with Pool(cpu_count()-6) as p:
        #     successes = list(tqdm(p.imap_unordered(self.run_fse_mpc, test_cases), total=len(test_cases)))

        print("Successful:")
        print(successes)


if __name__ == "__main__":
    fse_mpc = FEMPControl()
    fse_mpc.main()
