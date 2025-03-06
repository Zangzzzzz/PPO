import itertools
import math
import numpy as np
import sys
import matplotlib
from gym import spaces

matplotlib.use('Agg')
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk  # 本质上是对 Tcl/Tk 软件包的 Python 接口封装,

B_total = 500e6  # 总带宽500MHz
B_ = 125e6  # 单个子信道单宽 62.5MHz
N0 = 4e-21  # 噪声功率谱密度
high_star = 35786000
c = 3e8
num_power_channels = 4
total_power = 500.0  # 总功率
num_beams = 4
num_areas = 7
# 定义七个小区的编号
all_areas = [1, 2, 3, 4, 5, 6, 7]
# 计算所有可能的小区组合
num_power_groups = 35
areas = list(range(1, num_areas + 1))
possible_combinations = [sum(2 ** (area - 1) for area in combination) for combination in
                         itertools.combinations(areas, 4)]
all_action = []


def abs_sum(L):
    return np.sum(np.abs(L))


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        power_action_space = spaces.Box(low=1, high=total_power, shape=(num_power_channels,))
        action_spaces = [spaces.Discrete(35)] + [power_action_space] * 35
        self.action_space = spaces.Tuple(action_spaces)
        self.state_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,), dtype=float)

    def reset(self):
        self.update()
        # 使用七个小区初始化环境
        self.Qr = [0] * 7
        # 返回观察值
        return np.array(self.Qr)

    def step(self, action):
        # 分离波束和功率动作
        # 解释动作
        h = [0] * 7
        selected_areas = self.decode_action(possible_combinations[action[0]])
        # 4 5 6 7
        # 功率分配
        power_allocation = action[1]
        total_allocated_power = np.sum(power_allocation)
        scaling_factor = min(total_power / total_allocated_power, 1.0)
        scaled_power_allocation = power_allocation * scaling_factor
        # 存储每个小区的需求和提供量
        demand_and_provision = []
        differences = []
        all_demands = [self.get_demand(area) for area in all_areas]
        # 计算每个小区的提供量
        for area, power in zip(selected_areas, scaled_power_allocation):
            demand = all_demands[area - 1]  # 从环境中获取小区的需求
            provision = self.calculate_provision(area, power)  # 根据小区和功率计算提供量
            demand_and_provision.append((demand, provision))
            diff = max(0, (provision - demand))
            all_demands[area - 1] = diff
            differences.append(diff)
        k = sum(abs(value) for value in all_demands)
        # print(k)
        reward = 1 / (1 + np.exp(k - 40000))

        next_state = h

        return next_state, reward

    def decode_action(self, action_index):
        binary_representation = bin(int(action_index))[2:].zfill(7)
        selected_areas = [i + 1 for i, bit in enumerate(binary_representation) if bit == '1']
        return selected_areas

    def get_demand(self, area):
        Qr = [5639, 11750, 29458, 18125, 5482, 8841, 14733]
        x = Qr[area - 1]
        return x

    def calculate_provision(self, area, power):
        P = [(721, 1193), (203, 1178), (587, 528), (153, 743), (201, 276), (951, 119), (1146, 535)]
        S = (750, 750)
        x = [0] * 7
        y = [0] * 7
        for i in range(7):
            x[i] = np.linalg.norm(np.array(P[i]) - np.array(S))
            y[i] = math.sqrt(x[i] ** 2 + high_star ** 2)
        Gt = [104843.88829374622,
              104890.5995949591,
              38348.54344961949,
              79758.76249649755,
              104897.75268634477,
              104883.4226769876,
              104844.86251604515,
              ]
        # Gt = 158489.319  # 发射天线增益,52dBi
        Gr = 158489.319  # 接收天线增益
        yibusheluo = 2.000e10
        Ls = (c / (4 * math.pi * y[area - 1] * yibusheluo)) ** 2
        a = np.random.lognormal(0.5, 0.3)
        a1 = round(a, 1)  # 保留一位小数
        a2 = a1 / 10
        a3 = pow(10, a2 * 100)
        a4 = pow(a3, 1 / 100)
        Lp = 1 / a4
        h = Gt[area - 1] * Gr * Ls * Lp
        z = (power * h) / (N0 * B_)
        D = int((B_ * math.log(abs(1 + z), 2)) * 1e-5)
        return D

    # def render(self):
    #     # time.sleep(0.01)
    #     self.update()
# # 测试环境
# env = Maze()
# state = env.reset()
# action = env.action_space.sample()  # 随机选择一个动作
# next_state, reward = env.step(action)
#
# # 打印测试结果
# print("State:", state)
# print("Action:", action)
# print("Next State:", next_state)
# print("Reward:", reward)
