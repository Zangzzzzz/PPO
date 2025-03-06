import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1e5):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state, done = zip(*batch)
        state, action, reward, next_state = zip(*batch)
        # state, action, reward, next_state, done = map(np.array, zip(*batch))
        # return state, action, reward, next_state, done
        return state, action, reward, next_state

    # def push(self, state, action_with_param, reward, next_state, done):
    def push(self, state, action_with_param, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[int(self.position)] = (state, action_with_param, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)