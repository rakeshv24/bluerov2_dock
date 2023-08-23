import os
import time
import numpy as np
import sys
import rospy

# sys.path.insert(0, '/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/src/bluerov2_dock')

from casadi import evalf
from auv_hinsdale import AUV
from mpc_hinsdale import MPC


class MPControl():
    def __init__(self):
        cwd = os.path.dirname(__file__)
            
        # auv_yaml = cwd + "/../../config/auv_bluerov2.yaml"
        # mpc_yaml = cwd + "/../../config/mpc_bluerov2.yaml"
                    
        auv_yaml = cwd + "/../../config/auv_bluerov2_heavy.yaml"
        mpc_yaml = cwd + "/../../config/mpc_bluerov2_heavy.yaml"
        
        # auv_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/auv_bluerov2.yaml")
        # mpc_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/mpc_bluerov2.yaml")
        
        # auv_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/auv_bluerov2_heavy.yaml")
        # mpc_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/mpc_bluerov2_heavy.yaml")

        # Change these values as desired
        self.tolerance = 0.25
        self.yaw_tolerance = 0.05
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
        
        self.opt_data = {}
            
    def wrap_pi2negpi(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def wrap_zeroto2pi(self, angle):
        return angle % (2 * np.pi)
    
    def run_mpc(self, x0, xr):
        process_t0 = time.perf_counter()
        self.distance = np.linalg.norm(x0[0:3, :] - xr[0:3, :])
        
        x0[3:6, :] = self.wrap_pi2negpi(x0[3:6, :])
        # xr[5, :] += np.pi
        xr[3:6, :] = self.wrap_pi2negpi(xr[3:6, :])
        
        self.yaw_diff = abs((((x0[5, :] - xr[5, :]) + np.pi) % (2*np.pi)) - np.pi)[0]

        if self.distance < self.tolerance and self.yaw_diff < self.yaw_tolerance:
            # return np.zeros((8,1)), np.zeros((8,1)), True
            return np.zeros((8,1)), True
        
        else:    
            u, inst_cost = self.mpc.run_mpc(x0, xr)
            
            # x_dot = self.auv.compute_nonlinear_dynamics(x0, u, complete_model=True)
            
            # x_dot = np.array(evalf(x_dot))

            # x_sim = x0 + x_dot[0:12, :] * self.dt
            # x_sim[3:6, :] = self.wrap_pi2negpi(x_sim[3:6, :])
            
            # self.path_length += np.linalg.norm(x_sim[0:3, 0] - x0[0:3, 0])
            # self.distance = np.linalg.norm(x_sim[0:3, :] - xr[0:3, :])
            # self.yaw_diff = abs((((x_sim[5, :] - xr[5, :]) + np.pi) % (2*np.pi)) - np.pi)[0]
            
            self.comp_time = time.perf_counter() - process_t0
            
            print(f"T = {round(self.t_span[self.time_id],3)}s, Time Index = {self.time_id}")
            print(f"Computation Time = {round(self.comp_time,3)}s")
            print("----------------------------------------------")
            print(f"MPC Contol Input: {np.round(u, 2).T}")
            print("----------------------------------------------")
            print(f"Initial Vehicle Pose: {np.round(x0[0:6], 3).T}")
            print(f"Initial Vehicle Velocity: {np.round(x0[6:12], 3).T}")
            # print("----------------------------------------------")
            # print(f"Sim Vehicle Pose: {np.round(x_sim[0:6], 3).T}")
            # print(f"Sim Vehicle Velocity: {np.round(x_sim[6:12], 3).T}")
            print("----------------------------------------------")
            print(f"Dock Pose: {np.round(xr[0:6], 3).T}")
            print(f"Dock Velocity: {np.round(xr[6:12], 3).T}")
            print("----------------------------------------------")
            print(f"Path length: {np.round(self.path_length, 3)}")
            print(f"(Dock-AUV) Distance to go: {np.round(self.distance, 3)}")
            print(f"(Dock-AUV) Yaw difference: {np.round(self.yaw_diff, 3)}")
            print("----------------------------------------------")
            # print("")
            
            # x_sim[11, :] = np.round(x_sim[11, :], 2)
            
            self.time_id += 1
                    
            # return u, x_sim, False
            return u, False


if __name__ == "__main__":
    mpc = MPControl()
