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

def select_action(net: DQNNet, board, eps: float, device: str = "cpu") -> int:
    """ε-greedy：eps 内随机合法位，否则对 mask 后的 Q 值 argmax（返回 int 动作）。"""
    legal = np.flatnonzero(board.valid_moves())
    if np.random.random() < eps:
        return int(np.random.choice(legal))
    with torch.no_grad():
        obs = torch.as_tensor(board.observation(), dtype=torch.float32).unsqueeze(0)
        q = net(obs.to(device)).squeeze(0)               # (225,)
    q = q.detach().cpu().numpy()
    q[np.array([i for i in range(SIZE * SIZE) if i not in legal.tolist()])] = -np.inf
    return int(legal[q[legal].argmax()])


class DQNTrainer:
    """DQN 训练核心：replay 采样 → TD 靶子（target 网络 + 合法 mask）→ MSE → 一步 Adam。

    buffer 默认 50 万条；target 每 target_sync_steps 次 train_step 同步一次。
    """

    def __init__(self, net: DQNNet, gamma: float = 0.99, lr: float = 1e-4,
                 batch_size: int = 256, target_sync_steps: int = 10_000,
                 device: str = "cpu"):
        self.net = net.to(device)
        self.target_net = DQNNet(net.size, net.features[0].out_channels).to(device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_steps = target_sync_steps
        self.device = device
        self.steps_done = 0

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())

    def get_eps(self, episode: int, eps_start: float = 1.0,
                eps_end: float = 0.05, eps_decay: int = 720_000) -> float:
        frac = min(1.0, episode / max(1, eps_decay))
        return eps_start + (eps_end - eps_start) * frac

    @staticmethod
    def _legal_mask(obs_batch: torch.Tensor) -> torch.Tensor:
        """由 (B,2,size,size) 观测推导合法位 mask：两平面皆 0 = 空位。"""
        return (obs_batch[:, 0] + obs_batch[:, 1]) == 0   # (B,size,size) bool

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, sn, d = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, device=self.device)
        sn = torch.as_tensor(sn, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)
        q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)   # (B,)
        with torch.no_grad():
            q_next = self.target_net(sn)                        # (B,225)
            legal = self._legal_mask(sn).reshape(self.batch_size, -1)
            q_next = q_next.masked_fill(~legal, -1e9)           # 非法位不入 max
            max_q = q_next.max(dim=1).values
            target = r + self.gamma * max_q * (1.0 - d)          # done 只取 reward
        loss = torch.nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1
        if self.steps_done % self.target_sync_steps == 0:
            self.sync_target()
        return float(loss.item())