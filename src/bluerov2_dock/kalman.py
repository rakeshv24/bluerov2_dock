import numpy as np
import numpy.linalg as LA


class KalmanFilter():
    def __init__(self, f1_state, Q, R, F_k, H_k):
        self.Q = Q
        self.R = R
        self.F_k = F_k
        self.H_k = H_k
        self.x = f1_state
        self.P = 1e-6 * np.eye(f1_state.shape[0])

    def predict(self, chi):
        self.F = self.F_k(chi)
        self.x = self.F @ self.x
        self.P = (self.F @ self.P @ self.F.T) + self.Q
        
        state_pred = self.x
        cov_pred = self.P
        return state_pred, cov_pred

    def update(self, chi, z, P_auv_pred):
        R_pf = np.zeros((4, 4))
        R_pf[0:3, 0:3] = P_auv_pred[9:12, 9:12]
        R_pf[3, 3] = P_auv_pred[8, 8]
        self.H = self.H_k(chi)
        self.y = z - (self.H @ self.x)
        self.S = (self.H @ self.P @ self.H.T) + (self.R + R_pf)
        self.K = (self.P @ self.H.T @ LA.inv(self.S))
        I = np.eye(self.x.shape[0])

        self.x = self.x + (self.K @ self.y)
        self.P = (I - (self.K @ self.H)) @ self.P
        # self.P = ((I - (self.K @ self.H)) @ self.P @
        #           (I - (self.K @ self.H)).T) + (self.K @ self.R @ self.K.T)
        
        state_est = self.x
        cov_est = self.P
        return state_est, cov_est