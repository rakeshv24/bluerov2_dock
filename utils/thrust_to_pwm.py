import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sympy as sp
from sympy.abc import x


csv = pd.read_csv("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/data/T200_data_16V.csv")

thrust_vals = csv['Force'].tolist()
neg_thrust = [i for i in thrust_vals if i < 0]
pos_thrust = [i for i in thrust_vals if i > 0]
zero_thrust = [i for i in thrust_vals if i == 0]

pwm_vals = csv['PWM'].tolist()
neg_t_pwm = [pwm_vals[i] for i in range(len(neg_thrust))]
zero_t_pwm = [pwm_vals[i] for i in range(len(neg_thrust), len(neg_thrust)+len(zero_thrust))]
pos_t_pwm = [pwm_vals[i] for i in range(len(neg_thrust)+len(zero_thrust), len(thrust_vals))]

pwm_arr = np.array(pwm_vals)
thrust_arr = np.array(thrust_vals)

neg_t = np.array(neg_thrust)
pos_t = np.array(pos_thrust)
neg_pwm = np.array(neg_t_pwm)
pos_pwm = np.array(pos_t_pwm)

coeff = np.polyfit(thrust_vals, pwm_vals, 5)
ffit = np.polyval(coeff, thrust_vals)

# sp.init_printing()
# print(sp.Poly(np.poly1d(coeff).coef,x).as_expr())
# sp.Poly(np.poly1d(coeff).coef,x).as_expr()

# plt.plot(thrust_vals, ffit)
# plt.plot(thrust_vals, pwm_vals)
# plt.show()
