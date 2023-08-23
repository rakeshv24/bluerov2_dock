import numpy as np
import numpy.linalg as LA
from casadi import SX, Function, evalf, jacobian, substitute


class ExtendedKalmanFilter():
    def __init__(self, ini_state, Q, R, F_k, H_k):
        self.Q = Q
        self.R = R
        self.F_k = F_k
        self.H_k = H_k
        self.x = ini_state
        self.P = 1e-1 * np.eye(ini_state.shape[0])
        
        x_var = SX.sym('x_var', (12,1))
        H_var = SX.sym('H_var', (10,1))
        F_var = SX.sym('F_var', (12,1))
        
        self.chi_sym = SX.sym('chi', 12, 1)
        self.f_B_sym = SX.sym('f_B', 3, 1)
        self.nu_w_sym = SX.sym('nu_w', 3, 1)
        self.nu_w_dot_sym = SX.sym('nu_w_dot', 3, 1)
        self.u_sym = SX.sym('u', 6, 1)
        
        F_jac = self.linearization(F_var, x_var)
        H_jac = self.linearization(H_var, x_var)
        self.F_jac = Function('F_jac', [F_var, x_var], [F_jac])
        self.H_jac = Function('H_jac', [H_var, x_var], [H_jac])
        
        self.compute_jacobians()
        
    def compute_jacobians(self):
        self.F_dot = self.F_jac(self.F_k(self.chi_sym, self.u_sym, self.f_B_sym, self.nu_w_sym, self.nu_w_dot_sym), self.chi_sym)
        self.H_dot = self.H_jac(self.H_k(self.chi_sym, self.u_sym, self.f_B_sym, self.nu_w_sym, self.nu_w_dot_sym), self.chi_sym)

    def predict(self, chi, u, f_B, nu_w, nu_w_dot):
        self.F = evalf(self.F_k(chi, u, f_B, nu_w, nu_w_dot))
        F_dot = substitute(self.F_dot, self.chi_sym, chi)
        F_dot = substitute(F_dot, self.f_B_sym, f_B)
        F_dot = substitute(F_dot, self.nu_w_sym, nu_w)
        F_dot = substitute(F_dot, self.nu_w_dot_sym, nu_w_dot)
        F_dot = substitute(F_dot, self.u_sym, u)
        F_dot = evalf(F_dot)
                
        self.x = self.F
        self.P = (self.F_dot @ self.P @ self.F_dot.T) + self.Q
        
        state_pred = self.x
        cov_pred = self.P
        return state_pred, cov_pred

    def update(self, z, chi, u, f_B, nu_w, nu_w_dot):
        self.H = evalf(self.H_k(chi, u, f_B, nu_w, nu_w_dot))
        H_dot = substitute(self.H_dot, self.chi_sym, chi)
        H_dot = substitute(H_dot, self.f_B_sym, f_B)
        H_dot = substitute(H_dot, self.nu_w_sym, nu_w)
        H_dot = substitute(H_dot, self.nu_w_dot_sym, nu_w_dot)
        H_dot = substitute(H_dot, self.u_sym, u)
        H_dot = evalf(H_dot)
        
        self.y = z - self.H
        self.S = evalf(self.H_dot @ self.P @ self.H_dot.T) + self.R
        self.K = evalf(self.P @ self.H_dot.T @ LA.inv(self.S))
        I = np.eye(self.x.shape[0])

        self.x = self.x + (self.K @ self.y)
        self.P = (I - (self.K @ self.H_dot)) @ self.P
                  
        state_est = self.x
        cov_est = self.P
        return state_est, cov_est
    
    def linearization(self, f, x):
        f_dot = jacobian(f, x)
        return f_dot