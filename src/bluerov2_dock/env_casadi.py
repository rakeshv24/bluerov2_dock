import numpy as np
from casadi import SX, inv, skew


class EnvironmentManager():
    def __init__(self, auv):
        self.auv = auv
        self.constant_current = []

    def init_env(self, env_model, model_type, val):
        env = {"params":{},
               "grids":{}}

        env["model"] = model_type
        env["model_name"] = env_model
        env["update_t"] = np.inf
        
        if model_type == 'constant':
            self.constant_current = val
        elif model_type == 'time-varying':
            env["params"]["epsilon"] = 0.3
            env["params"]["A"] = val[0]
            env["params"]["period"] = val[1] * 3600
            env["params"]["omega"] = 2 * np.pi / env["params"]["period"]
            env["params"]["scale"] = 10e3
            env["params"]["translation"] = [-7.5e3, -7.5e3]
            env["grids"]["x_max"] = 2
            env["grids"]["y_max"] = 1

        return env

    def handle_env(self, env, t):
        env, status = self.check_env(env, t)
        if status == 1:
            env = self.update_env(env)
        return env

    def check_env(self, env, t):
        if "last_load_time" not in env:
            status = 1
        elif t - env["last_load_time"] > env["update_t"]:
            status = 2
        else:
            status = 0

        env["last_load_time"] = t
        return env, status

    def update_env(self, env):
        return env

    def get_env(self, eta, nu, env, t):
        c = None
        w = None
        env_flow = {}
        env_model = env["model"]

        if env_model == 'constant':
            env_flow['f'] = SX.zeros((3, 1))
            w = 0.0
            env_flow['f'][0,0] = self.constant_current[0]
            env_flow['f'][1,0] = self.constant_current[1]
            env_flow['f'][2,0] = w
        elif env_model == 'time-varying':
            if env['model_name'] == 'sine_flow':
                fx = env['params']['A'] * np.sin(env['params']['omega'] * t)
                fy = 0.0
                fz = 0.0
                env_flow['f'] = SX.zeros((3,1))
                env_flow['f'][0,0] = fx
                env_flow['f'][1,0] = fy
                env_flow['f'][2,0] = fz
            elif env['model_name'] == 'double_gyre' or env['model_name'] == 'double_gyre_fast':
                x = (eta[1] - env["params"]["translation"][0]) / env["params"]["scale"]
                y = (eta[0] - env["params"]["translation"][1]) / env["params"]["scale"]
                
                at = env["params"]["epsilon"] * np.sin(env["params"]["omega"] * t)
                bt = 1 - 2 * env["params"]["epsilon"] * np.sin(env["params"]["omega"] * t)
                fx = at * x**2 + bt * x
                
                dphi_dy = env["params"]["A"] * np.pi * np.sin(np.pi * fx) * np.cos(np.pi * y)
                dphi_dx = env["params"]["A"] * np.pi * (2 * at * x + bt) * np.cos(np.pi * fx) * np.sin(np.pi * y)
                
                fx = (-dphi_dy)
                fy = (dphi_dx)
                
                c = [fx, fy]
                w = 0.0
                
                env_flow['f'] = SX.zeros((3,1))
                env_flow['f'][0,0] = c[1]
                env_flow['f'][1,0] = c[0]
                env_flow['f'][2,0] = w
            
        else:
            print("Wrong ocean model")
        
        if eta.shape[0] == 6 and 'f_B' not in env_flow:
            env_flow['f_B'] = inv(self.auv.compute_transformation_matrix(eta)[0:3, 0:3]) @ env_flow['f']
            S = skew(nu[3:6, :])
            env_flow['f_B_dot'] = -S @ env_flow['f_B']

        return env_flow
            