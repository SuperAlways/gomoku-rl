"""课 01 · E1：15×15 完整配方 DQN。

三件套（DQNNet / ReplayBuffer / DQNTrainer）+ 自对弈收集 play_episode，
全部不依赖 Arena（带 DB 太重）；训练循环手攥 Board，与 tabular.py 同风格。
"""

import numpy as np
import torch
from torch import nn

from gomoku.core import BLACK, DRAW, WHITE

SIZE = 15
WIN_LEN = 5


class DQNNet(nn.Module):
    """小 CNN：2 特征平面 → 2×Conv64 → FC64 → 225 个 Q 值（不 mask，mask 在动作选择/靶子处做）。"""

    def __init__(self, size: int = SIZE, channels: int = 64):
        super().__init__()
        self.size = size
        self.num_actions = size * size
        self.features = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * size * size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(obs))

class ReplayBuffer:
    """环形经验回放：存 (state, action, reward, next_state, done)，满则覆盖最旧。"""

    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self.states = np.zeros((capacity, 2, SIZE, SIZE), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.next_states = np.zeros((capacity, 2, SIZE, SIZE), np.float32)
        self.dones = np.zeros(capacity, bool)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self) -> int:
        return self.size