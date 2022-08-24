import os
import time
import numpy as np
import sys

sys.path.insert(0, '/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/src/bluerov2_dock')

from casadi import evalf
from auv_hinsdale import AUV
from mpc_hinsdale import MPC
import pickle


class MPControl():
    def __init__(self):    
        auv_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/auv_bluerov2.yaml")
        mpc_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/mpc_bluerov2.yaml")

        # Change these values as desired
        self.tolerance = 0.25
        self.path_length = 0.0
        self.p_times = [0]
        
        self.auv = AUV.load_params(auv_yaml)
        self.mpc = MPC.load_params(auv_yaml, mpc_yaml)
        
        self.comp_time = 0.
        self.time_id = 0
        self.dt = self.mpc.dt
        self.t_f = 3600.0
        self.t_span = np.arange(0.0, self.t_f, self.dt)
        self.mpc.reset()
                
        self.wec_data = {"state": {}}
        self.wec_data["state"]["eta"] = np.zeros((6, len(self.t_span) + 1))
        self.wec_data["state"]["nu_r"] = np.zeros((6, len(self.t_span) + 1))
        
        self.nav_data = {"state":{},
                         "control":{},
                         "analysis":{}}
        self.nav_data["state"]["t"] = self.t_span
        self.nav_data["state"]["eta"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["state"]["nu_r"] = np.zeros((6, len(self.t_span) + 1))
        self.nav_data["control"]["u"] = np.zeros((self.mpc.thrusters, len(self.t_span)))
        self.nav_data["analysis"]["eta_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["nu_r_dot"] = np.zeros((6, len(self.t_span)))
        self.nav_data["analysis"]["inst_cost"] = np.zeros((1, len(self.t_span)))
        self.nav_data["analysis"]["thrust_force"] = np.zeros((6, len(self.t_span)))
        
        self.opt_data = {}
            
    def wrap_pi2negpi(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def run_mpc(self, x0, xr):
        process_t0 = time.perf_counter()
        self.distance = np.linalg.norm(x0 - xr)
        
        x0[3:6, :] = self.wrap_pi2negpi(x0[3:6, :])
        xr[5, :] += np.pi
        xr[3:6, :] = self.wrap_pi2negpi(xr[3:6, :])
        
        print("X0: {}".format(x0[0:6, :]))
        print("Xr: {}".format(xr[0:6, :]))

        if self.distance < self.tolerance:
            return np.zeros((8,1)), True
        
        u, inst_cost, thrust_force = self.mpc.run_mpc(x0, xr)
        # print(u)
        
        x_dot = self.auv.compute_nonlinear_dynamics(x0, u, complete_model=True)
        
        x_dot = np.array(evalf(x_dot))

        x_sim = x0 + x_dot[0:12, :] * self.dt
        x_sim[3:6, :] = self.wrap_pi2negpi(x_sim[3:6, :])
        
        self.path_length += np.linalg.norm(x_sim[0:3, 0] - x0[0:3, 0])
        self.distance = np.linalg.norm(x_sim[0:6, :] - xr[0:6, :])
        
        self.wec_data["state"]["eta"][:, self.time_id] = xr[0:6, :].flatten()
        self.wec_data["state"]["nu_r"][:, self.time_id] = xr[6:12, :].flatten()
        
        self.nav_data["state"]["eta"][:, self.time_id+1] = x_sim[0:6, :].flatten()
        self.nav_data["state"]["nu_r"][:, self.time_id+1] = x_sim[6:12, :].flatten()
        self.nav_data["control"]["u"][:, self.time_id] = u.flatten()
        self.nav_data["analysis"]["eta_dot"][:, self.time_id] = x_dot[0:6, :].flatten()
        self.nav_data["analysis"]["nu_r_dot"][:, self.time_id] = x_dot[6:12, :].flatten()
        self.nav_data["analysis"]["inst_cost"][:, self.time_id] = inst_cost
        self.nav_data["analysis"]["thrust_force"][:, self.time_id] = thrust_force.flatten()
        
        self.comp_time += time.perf_counter() - process_t0
        
        print(f"T = {round(self.t_span[self.time_id],3)}s, Time Index = {self.time_id}")
        print(f"Computation Time = {round(self.comp_time,3)}s")
        print("----------------------------------------------")
        print(f"MPC Contol Input: {np.round(u, 3).T}")
        print(f"Thrust Force: {np.round(thrust_force, 3).T}")
        print("----------------------------------------------")
        print(f"Initial Vehicle Pose: {np.round(x0[0:6], 3).T}")
        print(f"Initial Vehicle Velocity: {np.round(x0[6:12], 3).T}")
        print("----------------------------------------------")
        print(f"Sim Vehicle Pose: {np.round(x_sim[0:6], 3).T}")
        print(f"Sim Vehicle Velocity: {np.round(x_sim[6:12], 3).T}")
        print("----------------------------------------------")
        print("WEC Pose: ", self.wec_data["state"]["eta"][:, self.time_id].T)
        print("WEC Velocity: ", self.wec_data["state"]["nu_r"][:, self.time_id].T)
        print("----------------------------------------------")
        print(f"Path length: {np.round(self.path_length, 3)}")
        print(f"(WEC-AUV) Difference in state space: {np.round(self.distance, 3)}")
        print("----------------------------------------------")
        # print("")
        
        self.opt_data["comp_time"] = self.comp_time
        self.opt_data["path_length"] = self.path_length
        self.opt_data["opt_index"] = self.time_id
        self.opt_data["horizon"] = self.mpc.horizon
        self.opt_data["dt"] = self.mpc.dt
        self.opt_data["full_body"] = self.mpc.model_type
        
        self.time_id += 1
                
        return thrust_force, False
        
    def log_data(self, path):
        with open(path + '/opt_data.pkl', 'wb') as f:
            pickle.dump(self.opt_data, f)
        
        with open(path + '/sim_nav_data.pkl', 'wb') as f:
            pickle.dump(self.nav_data, f)
        
        with open(path + '/env_data.pkl', 'wb') as f:
            pickle.dump(self.env_data, f)
        
        with open(path + '/wec_data.pkl', 'wb') as f:
            pickle.dump(self.wec_data, f)
            