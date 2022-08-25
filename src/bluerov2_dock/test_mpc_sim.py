from auto_dock import MPControl
import numpy as np

mpc = MPControl()

x0 = np.array([[0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
xr = np.array([[2., 3., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T

converge_flag = False

while not converge_flag:
    thrust, x0, converge_flag = mpc.run_mpc(x0, xr)

print("Converged")
