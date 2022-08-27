from auto_dock import MPControl
import numpy as np

mpc = MPControl()

x0 = np.array([[1.0, 2.0, 0.0, 0.0, 0.0, -3.13, 0., 0., 0., 0., 0., 0.]]).T
xr = np.array([[1.0, 2.5, 0.0, 0.0, 0.0, -2.13, 0., 0., 0., 0., 0., 0.]]).T

# x0 = np.array([[1.64, -0.27, -0.29, 0.0, 0.0, 1.57, 0., 0., 0., 0., 0., 0.]]).T
# xr = np.array([[-1.16, -0.21, -0.59, 0.0, 0.0, 1.57, 0., 0., 0., 0., 0., 0.]]).T

converge_flag = False

while not converge_flag:
    thrust, x0, converge_flag = mpc.run_mpc(x0, xr)

print("Converged")
