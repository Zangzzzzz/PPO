import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from gym import spaces
from model import Q_Actor, ParamNet
from memory import ReplayBuffer


class RandomAgent:
    def __init__(self, action_space, seed):
        self.action_space = action_space
        self.num_actions = self.action_space.spaces[0].n
        self.np_random = np.random.RandomState(seed=seed)
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, 1 + self.num_actions)])
        random.seed(seed)
        self.action_parameter_min_numpy = np.concatenate(
            [action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_max_numpy = np.concatenate(
            [action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()

    def choose_action(self, state):
        action = self.np_random.choice(self.num_actions)
        all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                   self.action_parameter_max_numpy))

        offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
        action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters

    def update(self):
        pass

class PDQNAgent:
    def __init__(self, state_space, action_space, epsilon_start=0.99, epsilon_end=0.02, epsilon_decay=3000,
                 batch_size=32, gamma=0.99, replay_memory_size=1e6, actor_lr=1e-4, param_net_lr=1e-4,
                 actor_kwargs={}, param_net_kwargs={}, device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None, loss_function=F.smooth_l1_loss):
        if param_net_kwargs is None:
            param_net_kwargs = {}
        self.action_space = action_space
        self.state_space = state_space
        self.device = torch.device(device)
        self.seed = seed
        random.seed(self.seed)
        self.np_random = np.random.RandomState(seed=seed)
        self.num_actions = action_space.spaces[0].n
        # print(self.num_actions)
        self.actions_count = 0
        # [4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4]
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, 1 + self.num_actions)])
        self.action_parameter_size = self.action_parameter_sizes.sum()   # 140
        self.action_parameter_max_numpy = np.concatenate(
            [self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [self.action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.epsilon = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.param_net_lr = param_net_lr
        self.actor_net = Q_Actor(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
                                 **actor_kwargs).to(self.device)
        self.actor_target = Q_Actor(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
                                    **actor_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor_net.state_dict())
        self.actor_target.eval()  # 不启用 BatchNormalization 和 Dropout
        self.param_net = ParamNet(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
                                  **param_net_kwargs).to(self.device)
        self.param_net_target = ParamNet(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
                                         **param_net_kwargs).to(self.device)
        self.param_net_target.load_state_dict(self.param_net.state_dict())
        self.param_net_target.eval()

        self.loss_func = loss_function
        self.actor_optimiser = optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.param_net_optimiser = optim.Adam(self.param_net.parameters(), lr=self.param_net_lr)

        self.memory = ReplayBuffer(capacity=replay_memory_size)

    def __str__(self):
        desc = "P-DQN Agent\n"
        desc += "Actor Network {}\n".format(self.actor_net) + \
                "Param Network {}\n".format(self.param_net) + \
                "Gamma:{}\n".format(self.gamma) + \
                "Batch Size {}\n".format(self.batch_size) + \
                "Seed{}\n".format(self.seed)

        return desc

    def choose_action(self, state, train=True):
        if train:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           math.exp(-1. * self.actions_count / self.epsilon_decay)
            self.actions_count += 1
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.param_net.forward(state)

                if random.random() < self.epsilon:
                    action = self.np_random.choice(self.num_actions)
                    all_action_parameters = torch.from_numpy(
                        np.random.uniform(self.action_parameter_min_numpy, self.action_parameter_max_numpy))
                else:
                    Q_value = self.actor_net.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    Q_value = Q_value.detach().data.numpy()
                    # print(Q_value.shape)
                    # action = Q_value.max(1)[1].item()
                    # action = Q_value.max(1).item()
                    action = np.argmax(Q_value, axis=1).item()

                all_action_parameters = all_action_parameters.cpu().data.numpy()
                # print(action)
                offset = np.array([self.action_parameter_sizes[i] for i in range(int(action))], dtype=np.int32).sum()
                action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.param_net.forward(state)
                Q_value = self.actor_net.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_value = Q_value.detach().data.numpy()
                # action = Q_value.max(1)[1].item()
                # action = Q_value.max(1).item()
                action = np.argmax(Q_value, axis=1).item()
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                offset = np.array([self.action_parameter_sizes[i] for i in range(int(action))], dtype=np.int32).sum()
                action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]
        # print('action, action_parameters, all_action_parameters')
        # print(action, action_parameters, all_action_parameters)

        return action, action_parameters, all_action_parameters

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        states = np.array(states)
        states = torch.from_numpy(states).to(self.device)
        actions = np.array(actions)
        actions_combined = torch.from_numpy(actions).to(self.device)
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = np.array(rewards)
        # rewards = torch.from_numpy(rewards).to(self.device).squeeze(1)
        rewards = np.expand_dims(rewards, axis=1)
        next_states = np.array(next_states)
        next_states = torch.from_numpy(next_states).to(self.device)
        # dones = np.array(dones)
        # dones = torch.from_numpy(dones).to(self.device).squeeze(1)
        # dones = np.expand_dims(dones, axis=1)
        rewards = torch.from_numpy(rewards).to(self.device)
        # dones = torch.from_numpy(dones).to(self.device)
        # -----------------------optimize Q actor------------------------
        with torch.no_grad():
            next_action_parameters = self.param_net_target.forward(next_states)
            q_value_next = self.actor_target(next_states, next_action_parameters)
            q_value_max_next = torch.max(q_value_next, 1, keepdim=True)[0].squeeze()

            # target = rewards + (1 - dones) * self.gamma * q_value_max_next
            target = rewards + self.gamma * q_value_max_next

        q_values = self.actor_net(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_actor = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_actor.backward()
        for param in self.actor_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.actor_optimiser.step()

        # ------------------------optimize param net------------------------------
        with torch.no_grad():
            action_params = self.param_net(states)
        action_params.requires_grad = True
        q_val = self.actor_net(states, action_params)
        param_loss = torch.mean(torch.sum(q_val, 1))
        self.actor_net.zero_grad()
        param_loss.backward()
        # todo parameter_loss design
# class PDQNAgent:
#     def __init__(self, state_space, action_space, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=5000,
#                  batch_size=32, gamma=0.90, replay_memory_size=1e6, actor_lr=1e-3, param_net_lr=1e-4,
#                  actor_kwargs={}, param_net_kwargs={}, device="cuda" if torch.cuda.is_available() else "cpu",
#                  seed=None, loss_function=F.smooth_l1_loss):
#         self.actions_count = 0
#         if param_net_kwargs is None:
#             param_net_kwargs = {}
#         self.action_space = action_space
#         self.state_space = state_space
#         self.device = torch.device(device)
#         self.seed = seed
#         random.seed(self.seed)
#         self.np_random = np.random.RandomState(seed=seed)
#         self.num_actions = action_space.spaces[0].n        # 35
#         self.actions_count = 0
#
#         # self.action_parameter_sizes = np.array(
#         #     [self.action_space.spaces[i].shape[0] for i in range(1, 1 + self.num_actions)])
#         # self.action_parameter_size = self.action_parameter_sizes.sum()
#         # self.action_parameter_max_numpy = np.concatenate(
#         #     [self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()
#         # self.action_parameter_min_numpy = np.concatenate(
#         #     [self.action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
#         # self.action_parameter_range = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
#
#
#         # 从第一个子空间获取信息
#         # if isinstance(action_space.spaces[0], spaces.Discrete):
#         #     self.action_parameter_sizes = np.array([action_space.spaces[0].n])
#         # elif isinstance(action_space.spaces[0], spaces.MultiDiscrete):
#         #     self.action_parameter_sizes = np.array(action_space.spaces[0].nvec)
#         # # 从第二个子空间获取信息（假设第二个子空间是连续功率分配）
#         # self.action_parameter_sizes = np.concatenate([
#         #     self.action_parameter_sizes,
#         #     np.array([action_space.spaces[1].shape[0]])
#         # ])
#         # self.action_parameter_size = self.action_parameter_sizes.sum()   #   39
#         self.action_parameter_size = 4
#         # print(self.action_parameter_size)
#         # 从第一个子空间获取参数范围
#         # self.action_parameter_max_numpy = np.concatenate([action_space.spaces[0].nvec]).ravel() if isinstance(
#         #     action_space.spaces[0], spaces.MultiDiscrete) else np.array([action_space.spaces[0].n])
#         # self.action_parameter_min_numpy = np.zeros_like(self.action_parameter_max_numpy)
#         # 从第二个子空间获取参数范围
#         self.action_parameter_max_numpy = np.concatenate([
#             # self.action_parameter_max_numpy,
#             action_space.spaces[1].high
#         ]).ravel()
#         self.action_parameter_min_numpy = np.concatenate([
#             # self.action_parameter_min_numpy,
#             action_space.spaces[1].low
#         ]).ravel()
#         self.action_parameter_range = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
#         # action_parameter_max_numpy = [100. 100. 100. 100.]
#         # action_parameter_min_numpy = [1. 1. 1. 1.]
#         # action_parameter_range = [99. 99. 99. 99.]
#
#
#         self.epsilon = 0
#         self.epsilon_start = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay = epsilon_decay
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.actor_lr = actor_lr
#         self.param_net_lr = param_net_lr
#
#         self.actor_net = Q_Actor(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
#                                  **actor_kwargs).to(self.device)
#         self.actor_target = Q_Actor(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
#                                     **actor_kwargs).to(self.device)
#         self.actor_target.load_state_dict(self.actor_net.state_dict())
#         self.actor_target.eval()  # 不启用 BatchNormalization 和 Dropout
#         self.param_net = ParamNet(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
#                                   **param_net_kwargs).to(self.device)
#         self.param_net_target = ParamNet(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
#                                          **param_net_kwargs).to(self.device)
#         self.param_net_target.load_state_dict(self.param_net.state_dict())
#         self.param_net_target.eval()
#
#         self.loss_func = loss_function
#         self.actor_optimiser = optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
#         self.param_net_optimiser = optim.Adam(self.param_net.parameters(), lr=self.param_net_lr)
#         self.memory = ReplayBuffer(capacity=replay_memory_size)
#
#     def __str__(self):
#         desc = "P-DQN Agent\n"
#         desc += "Actor Network {}\n".format(self.actor_net) + \
#                 "Param Network {}\n".format(self.param_net) + \
#                 "Gamma:{}\n".format(self.gamma) + \
#                 "Batch Size {}\n".format(self.batch_size) + \
#                 "Seed{}\n".format(self.seed)
#
#         return desc
#
#
#     def choose_action(self, state, train=True):
#         if train:
#             self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
#                            math.exp(-1. * self.actions_count / self.epsilon_decay)
#             self.actions_count += 1
#             with torch.no_grad():
#                 state = torch.tensor(state, device=self.device)
#                 state = state.view(1, -1)
#                 all_action_parameters = self.param_net.forward(state)
#
#                 # 随机选择动作和参数
#                 if random.random() < self.epsilon:
#                     action = self.np_random.choice(self.num_actions)
#                     all_action_parameters = torch.from_numpy(
#                         np.random.uniform(self.action_parameter_min_numpy, self.action_parameter_max_numpy))
#                 else:
#                     # 根据 Q 值选择离散动作
#                     Q_values = self.actor_net.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
#                     Q_values = Q_values.detach().data.numpy()
#                     # 获取 Q 值最大的离散动作索引
#                     action = Q_values.argmax(1).item()
#                     # action = Q_values.max(1)[1].item()
#                     # 获取连续动作参数
#                 all_action_parameters = all_action_parameters.cpu().data.numpy()
#                 # all_action_parameters = all_action_parameters[0]
#                 print('1')
#                 # offset = self.num_actions  # 离散动作的数量
#                 # action_parameters = all_action_parameters[0, offset:].cpu().data.numpy()
#
#         else:
#             with torch.no_grad():
#                 state = torch.tensor(state, device=self.device)
#                 all_action_parameters = self.param_net.forward(state)
#                 # 根据 Q 值选择离散动作
#                 Q_values = self.actor_net.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
#                 Q_values = Q_values.detach().data.numpy()
#                 # 使用 argmax 获取 Q 值最大的离散动作索引
#                 action = Q_values.argmax(1).item()
#                 # action = Q_values.max(1)[1].item()
#                 # 获取连续动作参数
#                 all_action_parameters = all_action_parameters.cpu().data.numpy()
#                 # all_action_parameters = all_action_parameters[0]
#                 print('2')
#                 # offset = self.num_actions  # 离散动作的数量
#                 # action_parameters = all_action_parameters[0, offset:].cpu().data.numpy()
#         #
#         # print('action')
#         # print(action)
#         # print('action_parameters')
#         # print(action_parameters)
#         print('all_action_parameters')
#         print(all_action_parameters)
#         # return action, action_parameters, all_action_parameters
#         return action, all_action_parameters
#     def update(self):
#         if len(self.memory) < self.batch_size:
#             return
#         states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
#         print(states, actions, rewards, next_states)
#         # states = torch.from_numpy(states).to(self.device)
#         # # print("action shape:", actions.shape)
#         # # print(actions)
#         # actions_combined = torch.from_numpy(actions).to(self.device)
#         # print("Actions combined shape:", actions_combined.shape)
#         # actions = actions_combined[:, 0].long()
#         # action_parameters = actions_combined[:, 1:]
#         # print("Action parameters shape:", action_parameters.shape)
#         # rewards = torch.from_numpy(rewards).to(self.device).squeeze(1)
#         # next_states = torch.from_numpy(next_states).to(self.device)
#         # dones = torch.from_numpy(dones).to(self.device).squeeze(1)
#         action_parameters = actions[1:]
#
#         # -----------------------optimize Q actor------------------------
#         with torch.no_grad():
#             next_action_parameters = self.param_net_target.forward(next_states)
#             q_value_next = self.actor_target(next_states, next_action_parameters)
#             q_value_max_next = torch.max(q_value_next, 1, keepdim=True)[0].squeeze()
#             # target = rewards + (1 - dones) * self.gamma * q_value_max_next
#             target = rewards + self.gamma * q_value_max_next
#
#         q_values = self.actor_net(states, action_parameters)
#         y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
#         y_expected = target
#         loss_actor = self.loss_func(y_predicted, y_expected)
#
#         self.actor_optimiser.zero_grad()
#         loss_actor.backward()
#         for param in self.actor_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
#         self.actor_optimiser.step()
#
#         # ------------------------optimize param net------------------------------
#         with torch.no_grad():
#             action_params = self.param_net(states)
#         action_params.requires_grad = True
#         q_val = self.actor_net(action_params)
#         param_loss = torch.mean(torch.sum(q_val, 1))
#         self.actor_net.zero_grad()
#         param_loss.backward()
#         # todo parameter_loss design