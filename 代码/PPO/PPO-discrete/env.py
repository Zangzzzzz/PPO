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
from itertools import permutations

B_ = 30e6  # 单个子信道单宽 25MHz
N0 = 4e-21  # 噪声功率谱密度
high_star = 35786000
c = 3e8
total_power = 125.0  # 最大功率
power = total_power / 5
num_subchannels = 5
all_action = []
def abs_sum(L):
    return np.sum(np.abs(L))

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = spaces.Discrete(120)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=float)

    def reset(self):
        self.update()
        # 使用七个小区初始化环境
        self.i = [0] * 5
        # 返回观察值
        return np.array(self.i)

    def step(self, action):
        # 解释动作
        # 子信道选择
        m = [0] * 5
        channel_allocation = action % 120
        all_channel = list(permutations(range(num_subchannels), num_subchannels))
        channel_group = all_channel[channel_allocation]
        # 功率分配
        m[:5] = channel_group
        # 存储每个小区的需求和提供量
        # all_demands = [2746, 2840, 3069, 3006, 3072]
        # 2大
        all_demands = [3046, 3140, 3369, 3306, 3372]
        # all_demands = [3746, 3840, 4069, 4006, 4072]
        # 2小
        # all_demands = [2446, 2540, 2769, 2706, 2772]

        all_provision = []
        # 计算每个小区的提供量
        i = 0
        for channel in channel_group:
            provision = self.calculate_provision(i, channel, power)  # 根据小区和功率计算提供量
            all_provision.append(provision)
            i = i+1
        all_provision = np.array(all_provision)  # 转换为NumPy数组
        diff = np.abs(all_provision - all_demands)
        k = sum(abs(value) for value in diff)
        reward = sum(abs(value) for value in all_provision)
        next_state = m
        return next_state, reward, k
    def power(self, a):
        power = (total_power/10)*a
        return power
    def calculate_provision(self, j, channel, power):
        P = [[1262, 508], [1189, 447], [1030, 562], [1114, 540], [1112, 560]]
        S = (1146, 535)
        x = [0] * 5
        y = [0] * 5
        for i in range(5):
            x[i] = np.linalg.norm(np.array(P[i]) - np.array(S))
            y[i] = math.sqrt(x[i] ** 2 + high_star ** 2)
        Gt = [67830.67985945876,
              78098.01296022063,
              67825.90679009916,
              101521.9104090984,
              99265.75807196557]
        # Gt = 158489.319  # 发射天线增益,52dBi
        Gr = 158489.319  # 接收天线增益
        yibusheluo = [1.980e10, 1.990e10, 2.000e10, 2.010e10, 2.020e10]
        yibu = yibusheluo[channel]
        Ls = (c / (4 * math.pi * y[j] * yibu)) ** 2
        a = np.random.lognormal(0.5, 0.3)
        a1 = round(a, 1)  # 保留一位小数
        a2 = a1 / 10
        a3 = pow(10, a2 * 100)
        a4 = pow(a3, 1 / 100)
        Lp = 1 / a4
        h = Gt[j] * Gr * Ls * Lp
        z = (power * h) / (N0 * B_)
        D = int((B_ * math.log(abs(1 + z), 2)) * 1e-5)
        return D
# 测试环境
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