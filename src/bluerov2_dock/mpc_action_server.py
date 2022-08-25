import rospy
import actionlib
from bluerov2_dock.msg import MPCFeedback, MPCResult, MPCAction

import os
import time
import numpy as np
import sys
import rospy

sys.path.insert(0, '/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/src/bluerov2_dock')

from casadi import evalf
from auv_hinsdale import AUV
from mpc_hinsdale import MPC
import pickle


class AutoControl():
    _feedback = MPCFeedback()
    _result = MPCResult()
    
    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, MPCAction, execute_cb=self.run_mpc, auto_start = False)
        self._as.start()
            
        auv_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/auv_bluerov2.yaml")
        mpc_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/mpc_bluerov2.yaml")
        
        # auv_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/auv_bluerov2_heavy.yaml")
        # mpc_yaml = os.path.join("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/mpc_bluerov2_heavy.yaml")

        # Change these values as desired
        self.tolerance = 0.10
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
                    
    def wrap_pi2negpi(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def run_mpc(self, goal):
        rate = rospy.Rate(10)
        
        x0 = goal.x0
        xr = goal.xr
        
        process_t0 = time.perf_counter()
        self.distance = np.linalg.norm(x0[0:6, :] - xr[0:6, :])
        
        x0[3:6, :] = self.wrap_pi2negpi(x0[3:6, :])
        # xr[5, :] += np.pi
        xr[3:6, :] = self.wrap_pi2negpi(xr[3:6, :])

        if self.distance < self.tolerance:
            self._result.converge_flag = True
            rospy.loginfo("[AutoControl][run_mpc] Reached target!")
            # self._as.
        
        u, _, thrust_force = self.mpc.run_mpc(x0, xr)
        
        self.comp_time = time.perf_counter() - process_t0
        
        rospy.loginfo(f"[AutoControl][run_mpc] T = {round(self.t_span[self.time_id],3)}s, Time Index = {self.time_id}")
        rospy.loginfo(f"[AutoControl][run_mpc] Computation Time = {round(self.comp_time,3)}s")
        rospy.loginfo("----------------------------------------------")
        rospy.loginfo(f"[AutoControl][run_mpc] MPC Contol Input: {np.round(u, 3).T}")
        rospy.loginfo(f"[AutoControl][run_mpc] Thrust Force: {np.round(thrust_force, 3).T}")
        rospy.loginfo("----------------------------------------------")
        
        self.time_id += 1
        
        
        
