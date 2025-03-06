import itertools
import math
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk  # 本质上是对 Tcl/Tk 软件包的 Python 接口封装,
k = 100
reward = 1 / (1 + np.exp(k - 60000))
print(reward)
# B_total = 500e6  # 总带宽500MHz
# B_ = 125e6  # 单个子信道单宽 62.5MHz
# N0 = 4e-21  # 噪声功率谱密度
# high_star = 35786000
# c = 3e8
# num_power_channels = 4
# total_power = 100.0  # 总功率
# Qr = [5639, 11750, 29458, 18125, 5482, 8841, 14733]
# demand_and_provision = []
#
# def calculate_provision(area, power):
#     P = [(721, 1193), (203, 1178), (587, 528), (153, 743), (201, 276), (951, 119), (1146, 535)]
#     S = (750, 750)
#     x = [0] * 7
#     y = [0] * 7
#     for i in range(7):
#         x[i] = np.linalg.norm(np.array(P[i]) - np.array(S))
#         y[i] = math.sqrt(x[i] ** 2 + high_star ** 2)
#     Gt = [104843.88829374622,
#           104890.5995949591,
#           38348.54344961949,
#           79758.76249649755,
#           104897.75268634477,
#           104883.4226769876,
#           104844.86251604515,
#           ]
#     # Gt = 158489.319  # 发射天线增益,52dBi
#     Gr = 158489.319  # 接收天线增益
#     yibusheluo = 2.000e10
#     Ls = (c / (4 * math.pi * y[area - 1] * yibusheluo)) ** 2
#     a = np.random.lognormal(0.5, 0.3)
#     a1 = round(a, 1)  # 保留一位小数
#     a2 = a1 / 10
#     a3 = pow(10, a2 * 100)
#     a4 = pow(a3, 1 / 100)
#     Lp = 1 / a4
#     h = Gt[area - 1] * Gr * Ls * Lp
#     z = (power * h) / (N0 * B_)
#     D = int((B_ * math.log(abs(1 + z), 2)) * 1e-5)
#     return D
#
# area = 7
# power = 100
# provision = calculate_provision(area, power)  # 根据小区和功率计算提供量
# demand = Qr[area - 1]  # 从环境中获取小区的需求
# diff = abs(provision - demand)
# print(provision)
# print(diff)




