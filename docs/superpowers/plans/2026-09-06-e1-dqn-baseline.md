# E1 完整配方 DQN 基线（A2 快门派）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 15×15 上从零训练一个完整配方的 DQN（小 CNN · replay buffer · target network · ε 退火 · 稀疏终局奖励 · 对手池 B2），产出"贴地 → 爬升"的训练曲线，作为课 01 后续 E2-E11 的对照基线。

**Architecture:** `src/gomoku/rl/dqn.py` 提供纯 torch 的三件套（`DQNNet` 小 CNN、`ReplayBuffer`、`DQNTrainer` 含 target 与 ε 退火）+ 一个与既有 `tabular.py` 同风格的 `play_episode`（学员恒执黑、对手经回调、终局回填稀疏奖励）。`scripts/train_dqn.py` 驱动 60 代训练、`scripts/evaluate_dqn.py` 做逐代基准赛，输出到 `runs/e1/`。训练循环直接手攥 `Board`（不经过带 DB 的 Arena，与 E0 一致）。

**Tech Stack:** Python 3.12 · numpy · PyTorch ≥2.7（本机 .venv 已装 cu128 GPU 版；测试在 CPU 上跑）· matplotlib（已有）

**Spec:** `docs/superpowers/specs/2026-09-06-e1-dqn-baseline-design.md`

## Global Constraints

- PyTorch 依赖：在 `pyproject.toml` 的 `dependencies` 加 `"torch>=2.7"`。本机 GPU 版经 cu128 index 安装；CPU wheels 从 PyPI 即可，测试全部在 CPU 上运行，不要求 GPU。
- torch 缺省时测试**跳过而非报错**：`test_dqn.py` 模块级 `pytest.importorskip("torch")`。
- 尺寸恒定：`SIZE=15, WIN_LEN=5`；输入特征平面 `(2,15,15)` float32（用 `Board.observation(BLACK)`）；动作 = `row*15+col`，共 225 个，非法点一律 mask、不进 max。
- 超参（spec 定稿，全部禁止在任务内改动）：帧期 2 万局 × 60 代 = 120 万局；γ=0.99；ε 1.0→0.05 线性、`eps_decay=720000` 局后钳在 0.05；纯稀疏 +1/-1/0；网络 2×Conv64→FC64→225；replay 50 万；target 每 1 万步同步；每 4 步学 1 次；batch=256；Adam `lr=1e-4`（E0 附录 `<a data-gist>`：函数近似下 lr 降到表格的 4 个数量级）。
- 对手池 B2：`random_opponent`（随机合法位）与 `MinimaxPlayer("easy")`（固定尺子）各 50% 随机抽、从头到尾交错，二者皆不可学习、不入学习。
- 逐代基准赛：每代后 vs random / vs minimax-easy 各 20 局，仅记录胜率，不动网络。
- 输出 `runs/e1/`，不入 git（现有规则）。
- 复用 `gomoku.core` 的 `Board`（含 `observation/valid_moves/action_to_move/play/current_player/game_over/winner/history`），不修改 core / arena / minimax。

---

### Task 1: PyTorch 依赖 + 网络体 DQNNet

**Files:**
- Modify: `pyproject.toml`（dependencies 加 torch）
- Create: `src/gomoku/rl/dqn.py`（`DQNNet` 起步，后续任务在**同一文件**追加）
- Test: `tests/test_dqn.py`（到 Task 4 为止只测 dqn.py，本任务先建顶部的 importorskip 与 torch 导入）

**Interfaces:**
- Produces: `class DQNNet(nn.Module)`，`__init__(self, size: int = 15, channels: int = 64)`，`forward(self, obs: torch.Tensor) -> torch.Tensor`，输入 `(B,2,size,size)` float32，输出 `(B, size*size)` 未 mask 的 Q 值。
- Produces: 模块级常量 `SIZE = 15`, `WIN_LEN = 5`。

- [ ] **Step 1: 在 pyproject.toml 加 torch 依赖**

`pyproject.toml` 的 `dependencies` 末尾加一行：

```toml
    "torch>=2.7",
```

- [ ] **Step 2: 建 `src/gomoku/rl/dqn.py` 顶部 + DQNNet 最小实现**

首次写入本文件（后续任务在此追加）：

```python
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
```

- [ ] **Step 3: 建 `tests/test_dqn.py` 顶部 + 网络形状测试**

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")   # torch 未装则整模块跳过

from gomoku.rl.dqn import DQNNet, SIZE  # noqa: E402


def test_dqn_net_forward_shape():
    net = DQNNet(size=15, channels=8)
    obs = torch.zeros(4, 2, 15, 15)
    out = net(obs)
    assert out.shape == (4, 225)


def test_dqn_net_legal_positions_distinct():
    # 空盘上传一个两平面不一致的观测，输出对动作序号敏感（非全零）
    net = DQNNet(size=15, channels=8)
    obs = torch.zeros(1, 2, 15, 15)
    obs[0, 0, 7, 7] = 1.0                       # 自方一枚
    out = net(obs).squeeze(0)
    assert out.shape == (SIZE * SIZE,)
    assert (out != 0).any()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_dqn.py -v`
Expected: 2 passed（若 torch 缺失则 skip，但本机 .venv 已装，应通过）

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/gomoku/rl/dqn.py tests/test_dqn.py
git commit -m "feat: E1 DQN 网络体 DQNNet（2×Conv64→FC64→225）+ torch 依赖"
```

---

### Task 2: ReplayBuffer（50 万容量环形缓冲）

**Files:**
- Modify: `src/gomoku/rl/dqn.py`（追加 `ReplayBuffer`）
- Test: `tests/test_dqn.py`（追加）

**Interfaces:**
- Produces: `class ReplayBuffer`，`__init__(self, capacity: int = 500_000)`；`push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool)`；`sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`（返回 `(states, actions, rewards, next_states, dones)`）；`__len__(self) -> int`。
- Consumes: `state` / `next_state` 为 `(2,15,15)` float32 ndarray；`sample` 返回 states/next_states 为堆叠的 `(B,2,15,15)` float32，actions int64 `(B,)`，rewards float32 `(B,)`，dones bool `(B,)`。

- [ ] **Step 1: 写失败的采样/容量测试**

```python
from gomoku.rl.dqn import ReplayBuffer


def test_replay_buffer_capacity_wraps():
    buf = ReplayBuffer(capacity=10)
    for i in range(25):
        buf.push(np.zeros((2, 15, 15), np.float32), i, 1.0,
                 np.ones((2, 15, 15), np.float32), False)
    assert len(buf) == 10


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    for i in range(8):
        buf.push(np.full((2, 15, 15), i, np.float32), i, 0.5,
                 np.full((2, 15, 15), i + 1, np.float32), False)
    states, actions, rewards, next_states, dones = buf.sample(8)
    assert states.shape == (8, 2, 15, 15)
    assert actions.shape == (8,) and actions.dtype == np.int64
    assert rewards.shape == (8,) and rewards.dtype == np.float32
    assert next_states.shape == (8, 2, 15, 15)
    assert dones.shape == (8,) and dones.dtype == bool
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_dqn.py -v`
Expected: 这 2 个测试断言 `ReplayBuffer` not defined。

- [ ] **Step 3: 实现 ReplayBuffer**

在 `dqn.py` 追加：

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_dqn.py -v`
Expected: 前两个测试转绿，本任务加上步共 4 passed。

- [ ] **Step 5: Commit**

```bash
git add src/gomoku/rl/dqn.py tests/test_dqn.py
git commit -m "feat: E1 ReplayBuffer（50 万容量环形缓冲）"
```

---

### Task 3: ε-greedy 动作选择（合法位 mask）

**Files:**
- Modify: `src/gomoku/rl/dqn.py`（追加 `select_action`）
- Test: `tests/test_dqn.py`（追加）

**Interfaces:**
- Produces: `def select_action(net: DQNNet, board, eps: float, device: str = "cpu") -> int`——持 `board` 当前局，ε-greedy：`eps` 内随机合法位，否则对 mask 后的 Q 值 argmax，返回 `int` 动作。调用方需先 `board` 在其执黑回合时传入。
- Consumes: `DQNNet.forward`；`board.valid_moves()`（bool 网格）；`board.observation`。

- [ ] **Step 1: 写失败测试**

```python
from gomoku.core import Board
from gomoku.rl.dqn import select_action


def test_select_action_legal_and_masks_occupied():
    b = Board()
    b.play(7, 7)          # 黑
    b.play(7, 8)          # 白
    net = DQNNet(size=15, channels=8)
    # 把 (7,7) 对应动作的 Q 调成全网最高，但该位已占，必须被 mask 掉
    with torch.no_grad():
        net.head[-1].weight.zero_()
        net.head[-1].bias.zero_()
        net.head[-1].bias[7 * 15 + 7] = 10.0
        net.head[-1].bias[0] = 5.0            # 空位 (0,0) 次高
    a = select_action(net, b, eps=0.0)
    assert b.action_to_move(a) == (0, 0)      # (7,7) 已占，被迫选 (0,0)
    assert 0 <= a < 225


def test_select_action_eps_random_legal():
    b = Board()
    net = DQNNet(size=15, channels=8)
    for _ in range(30):
        a = select_action(net, b, eps=1.0)
        r, c = b.action_to_move(a)
        assert b.grid[r, c] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_dqn.py -v`
Expected: `select_action` not defined 等失败。

- [ ] **Step 3: 实现 select_action**

在 `dqn.py` 追加：

```python
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
```

> 说明：先造全量 225 mask 再取 `legal` 内 argmax，保证非法点不被选中；绝不选已占位。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_dqn.py -v`
Expected: 全部转绿（本任务后共 6 passed）。

- [ ] **Step 5: Commit**

```bash
git add src/gomoku/rl/dqn.py tests/test_dqn.py
git commit -m "feat: E1 ε-greedy 动作选择（合法位 mask，防选已占位）"
```

---

### Task 4: target 网络 + 批量训练步（TD 靶子 + MSE + Adam）

**Files:**
- Modify: `src/gomoku/rl/dqn.py`（追加 `DQNTrainer` 的第一个方法子集：`__init__`、`sync_target`、`_legal_mask`、`train_step`）
- Test: `tests/test_dqn.py`（追加）

**Interfaces:**
- Produces: `class DQNTrainer`，`__init__(self, net: DQNNet, gamma: float = 0.99, lr: float = 1e-4, batch_size: int = 256, target_sync_steps: int = 10_000, device: str = "cpu")`；`self.net` / `self.target_net` / `self.optimizer` / `self.buffer`（ReplayBuffer 默认 50 万）/ `self.steps_done: int`；`def sync_target(self)` 浅拷贝 `net` 权重到 `target_net`；`def train_step(self) -> float | None`（缓冲不足返回 None，否则做一次 Adam 更新，`self.steps_done += 1`，返回 loss）；`def get_eps(self, episode: int, eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: int = 720_000) -> float`。
- Consumes: `ReplayBuffer.sample`；`DQNNet.forward`。

- [ ] **Step 1: 写失败测试（target 同步 + 训练步收敛性 + ε 退火端点）**

```python
from gomoku.rl.dqn import DQNTrainer, ReplayBuffer


def _dummy_batch(buf, n=16, done=False):
    s = np.random.randn(n, 2, 15, 15).astype(np.float32)
    a = np.random.randint(0, 225, n)
    r = np.zeros(n, np.float32)
    sn = np.random.randn(n, 2, 15, 15).astype(np.float32)
    d = np.full(n, done, bool)
    return s, a, r, sn, d


def test_sync_target_copies_weights():
    net = DQNNet(size=15, channels=8)
    trainer = DQNTrainer(net, batch_size=16, device="cpu")
    w = trainer.net.head[-1].weight.data.clone()
    with torch.no_grad():
        trainer.target_net.head[-1].weight.fill_(0.0)
    trainer.sync_target()
    assert torch.equal(trainer.target_net.head[-1].weight.data, w)


def test_train_step_returns_loss_and_increments():
    net = DQNNet(size=15, channels=8)
    trainer = DQNTrainer(net, batch_size=16, device="cpu")
    for _ in range(16):
        trainer.buffer.push(*([np.zeros((2, 15, 15), np.float32), 0, 0.0,
                             np.zeros((2, 15, 15), np.float32), False]))
    loss = trainer.train_step()
    assert loss is not None and loss == loss           # 有限且非 NaN
    assert trainer.steps_done == 1


def test_eps_linear_then_clamped():
    trainer = DQNTrainer(DQNNet(size=15, channels=8), batch_size=16)
    assert trainer.get_eps(0, 1.0, 0.05, 100) == pytest.approx(1.0)
    mid = trainer.get_eps(50, 1.0, 0.05, 100)
    assert mid < 1.0 and mid > 0.05
    assert trainer.get_eps(200, 1.0, 0.05, 100) == pytest.approx(0.05)


def test_train_step_none_below_batch():
    trainer = DQNTrainer(DQNNet(size=15, channels=8), batch_size=16)
    assert trainer.train_step() is None
```

> 注：`test_train_step_returns_loss_and_increments` 手动往 buffer 塞 16 条占位经验——train_step 只需 shape 正确即可算出有限 loss；合法 mask 与收敛正确性由 Task 6 的端到端测试覆盖。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_dqn.py`
Expected: `DQNTrainer` not defined。

- [ ] **Step 3: 实现 DQNTrainer（本次加 __init__/sync_target/_legal_mask/train_step/get_eps）**

在 `dqn.py` 追加：

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_dqn.py`
Expected: 全绿（本任务后共 10 passed）。

- [ ] **Step 5: Commit**

```bash
git add src/gomoku/rl/dqn.py tests/test_dqn.py
git commit -m "feat: E1 DQNTrainer（target 网络 + TD 靶子 + ε 线性退火 + Adam 步）"
```

---

### Task 5: play_episode 自对弈收集（对手池 B2 + 稀疏奖励回填）

**Files:**
- Modify: `src/gomoku/rl/dqn.py`（追加 `random_opponent`、`make_minimax_opponent`、`policy_entropy_from_logits`、`play_episode`）
- Test: `tests/test_dqn.py`（追加）

**Interfaces:**
- Produces:
  - `def random_opponent(board, rng) -> int`——随机合法位。
  - `def make_minimax_opponent(level: str = "easy")` → 返回 `opponent_fn(board, rng) -> int`，调用 `MinimaxPlayer(level).select_move(board)`。
  - `def policy_entropy_from_logits(q: np.ndarray, legal: np.ndarray, tau: float = 0.3) -> float`——softmax(Q/τ) 在合法动作上的 Shannon 熵。
  - `def play_episode(board, choose_black, opponent_fn, learner) -> dict`——自对弈一局：`choose_black(board)->int` 决定学员（黑）每一手；`opponent_fn(board)` 决定白方；每手挖 `learner.train_step()`（返回 loss 或 None，None=缓冲不足没训）、把黑每手 `learner.buffer.push` 进回放。`learner` 是持有 `.train_step()` 与 `.buffer` 的最小对象（真实传 `DQNTrainer`；基准赛传 `_NoopTrainer`）。返回 dict `{"winner": int, "steps": int, "losses": list[float]}`。
  - `board`: 透传的 `Board`（默认由调用方构造 `Board(size=SIZE, win_len=WIN_LEN)`）。

> **实现契约**：`play_episode` 只对**黑**（学员）的回合收经验。转移 `(s,a,r,s',done)` 中，`a` 是黑的一手，`s`=该手前的 `observation()`，`s'`=**对手应手后**黑重新能落子时的 `observation()`（即下一黑回合状态）；奖励全 0 直到终局，终局回填 `+1/-1/0`（`done=True`）。黑一手即连五取胜 → 无对手应手，立即以 `done=True, r=+1` 闭合。

- [ ] **Step 1: 写失败测试（一个溜达局 + 稀疏奖励 + 对手池两个函数）**

```python
import numpy as np

from gomoku.core import Board
from gomoku.rl.dqn import (random_opponent, make_minimax_opponent,
                           policy_entropy_from_logits, play_episode)


def _legal_rng(board, rng):
    return int(np.random.choice(np.flatnonzero(board.valid_moves())))


def test_random_opponent_legal():
    b = Board()
    a = random_opponent(b, np.random.default_rng(0))
    assert b.grid[b.action_to_move(a)] == 0


def test_minimax_opponent_returns_legal():
    opp = make_minimax_opponent("easy")
    b = Board(); b.play(7, 7)
    a = opp(b, np.random.default_rng(0))
    assert b.grid[b.action_to_move(a)] == 0


def test_policy_entropy_uniform_and_peaked():
    q = np.zeros(SIZE * SIZE)
    legal = np.arange(SIZE * SIZE)
    assert policy_entropy_from_logits(q, legal) == pytest.approx(np.log(SIZE * SIZE), rel=1e-3)
    q2 = np.zeros(SIZE * SIZE); q2[112] = 10.0
    assert policy_entropy_from_logits(q2, legal) < 0.1


def test_play_episode_sparse_reward_terminal():
    b = Board(size=SIZE, win_len=WIN_LEN)
    def fake_black(board):
        return int(np.random.choice(np.flatnonzero(board.valid_moves())))
    def fake_opponent(board):
        return int(np.random.choice(np.flatnonzero(board.valid_moves())))
    # 假 learner：持有 buffer 与 train_step
    class Fake:
        buf = None
        def __init__(self):
            self.buf = type("B", (), {"push": lambda *a, **k: None})()
            self.buffer = self.buf
        def train_step(self):
            return None
    info = play_episode(b, fake_black, fake_opponent, Fake())
    assert info["steps"] == len(b.history) > 0
    assert info["winner"] in {1, 2, 3}
```

> 注：本测试只验证 play_episode 结构正确（走完一局、steps/winner 合理）。真正的学习闭环（loss 下降、贴地→爬升）由 Task 6 的端到端小规模冒烟覆盖。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_dqn.py`
Expected: `random_opponent` 等 not defined。

- [ ] **Step 3: 实现四个函数**

在 `dqn.py` 追加：

```python
def random_opponent(board, rng=None) -> int:
    return int(np.random.choice(np.flatnonzero(board.valid_moves())))


def make_minimax_opponent(level: str = "easy"):
    from gomoku.players.minimax import MinimaxPlayer   # 延迟导入避免循环
    player = MinimaxPlayer(level)
    def opponent_fn(board, rng=None) -> int:
        return player.select_move(board)
    return opponent_fn


def policy_entropy_from_logits(q: np.ndarray, legal: np.ndarray, tau: float = 0.3) -> float:
    """Q 值导出策略熵：softmax(Q/τ) 在合法动作上的 Shannon 熵（越小越决定）。"""
    z = q[legal] / tau
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def play_episode(board, choose_black, opponent_fn, learner) -> dict:
    """自对弈一局：黑=学员（choose/replay 学习），白=对手脚本。

    learner 需提供 .train_step()（每步尝试学习，返回 loss 或 None）与 .buffer（push 经验）。
    转移按"黑回合闭合"收：黑一手后，等对手应完（或黑自己连五获胜），
    用当时最新 observation 作 next_state 闭合。终局回填稀疏 +1/-1/0。
    """
    losses = []
    pending = None   # (state_before, action)
    while not board.game_over:
        if board.current_player == BLACK:
            state = board.observation(BLACK)
            action = choose_black(board)
            r, c = board.action_to_move(action)
            board.play(r, c)
            pending = (state, action)
        else:
            action = opponent_fn(board)
            r, c = board.action_to_move(action)
            board.play(r, c)
        l = learner.train_step()         # 每步尝试学习（缓冲不足则 None/跳过）
        if l is not None:
            losses.append(l)
        if pending is not None:
            if board.game_over:
                reward = {BLACK: 1.0, WHITE: -1.0, DRAW: 0.0}[board.winner]
                learner.buffer.push(*pending, reward, board.observation(BLACK), True)
                pending = None
            elif board.current_player == BLACK:
                learner.buffer.push(*pending, 0.0, board.observation(BLACK), False)
                pending = None
    return {"winner": int(board.winner), "steps": len(board.history),
            "losses": losses}
```

> 说明：`learner` 是"有 `.train_step()` + `.buffer`"的最小对象——真实训练传 `DQNTrainer`，基准赛传 `_NoopTrainer`（只测不训）。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_dqn.py`
Expected: 全绿（本任务后共 14 passed）。

- [ ] **Step 5: Commit**

```bash
git add src/gomoku/rl/dqn.py tests/test_dqn.py
git commit -m "feat: E1 play_episode 自对弈收集（对手池 B2 + 稀疏奖励回填）"
```

---

### Task 6: DQNTrainer 主循环 train() + 逐代基准赛 evaluate + 双脚本 + 端到端冒烟

**Files:**
- Modify: `src/gomoku/rl/dqn.py`（追加 `train()` 主循环 + `evaluate`/`load_net`/`plot_curves` 辅助，全部收在同一模块，避免 scripts 非包导致的 import 问题）
- Create: `scripts/train_dqn.py`（60 代驱动 + 画图）
- Create: `scripts/evaluate_dqn.py`（加载 checkpoint 跑基准赛）
- Test: `tests/test_dqn.py`（end-to-end 冒烟）+ `tests/test_evaluate.py`（测 `evaluate`）

**Interfaces:**
- Produces:
  - `def train(trainer: DQNTrainer, episodes_per_era: int = 20_000, num_eras: int = 60, eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: int = 720_000, seed: int = 0, device: str = "cpu", out_dir: str = "runs/e1", evaluate_fn=None, eval_n: int = 20) -> list[dict]`——逐代自对弈（B2 对手池 + ε 退火 + replay 累积）、逐代基准赛、写 `{out_dir}/metrics.jsonl`、存 `{out_dir}/checkpoints/gen{N:04d}.pt`；返回 metrics 行 list。
  - `def evaluate(net: DQNNet, opponent, n: int = 20, device: str = "cpu") -> float`——贪心（eps=0）打 `n` 局固定对手的胜率，**只测不训**（用 no-op train_step）。
  - `def load_net(checkpoint: str, size=SIZE, channels=64, device="cpu") -> DQNNet`
  - `def plot_curves(metrics: list[dict], out_path: str) -> None`——4 条逐代曲线（win_vs_random / win_vs_easy / steps / loss）落 PNG。
  - `train()` 回调 `evaluate_fn(trainer) -> dict`（基准赛），未传则用内置 vs random / vs minimax-easy 各 20 局的 `_default_evaluate`。
- Consumes: `DQNNet/ReplayBuffer/DQNTrainer/select_action/random_opponent/make_minimax_opponent/policy_entropy_from_logits/play_episode`（Task 1-5 产出）。

- [ ] **Step 1: 写端到端冒烟测试（小参数 2 代×小预算，验证主循环 + metrics 形状 + checkpoint 落盘）**

`tests/test_evaluate.py`（测 evaluate 只测不训）：

```python
import pytest

torch = pytest.importorskip("torch")

from gomoku.rl.dqn import DQNNet, evaluate, random_opponent


def test_evaluate_returns_winrate_bound():
    # 随机初始网络贪心对乱下，胜率应落在 [0,1]，且测试期间不训练网络
    net = DQNNet(size=15, channels=8)
    rate = evaluate(net, random_opponent, n=20, device="cpu")
    assert 0.0 <= rate <= 1.0
```

`tests/test_dqn.py`（追加端到端冒烟）：

```python
from gomoku.rl.dqn import DQNTrainer, DQNNet, train


def test_train_smoke_end_to_end(tmp_path):
    net = DQNNet(size=15, channels=8)
    trainer = DQNTrainer(net, batch_size=8, device="cpu")
    metrics = train(trainer, episodes_per_era=6, num_eras=2,
                    eps_decay=6, device="cpu", out_dir=str(tmp_path))
    assert len(metrics) == 2
    assert all({"era", "steps", "loss", "entropy", "eps",
                "win_vs_random", "win_vs_easy"} <= set(m) for m in metrics)
    ckpt = tmp_path / "checkpoints" / "gen0001.pt"
    assert ckpt.exists()
    (tmp_path / "metrics.jsonl").exists()
```

> train() 的 batch_size 由 DQNTrainer 持有，train() 签名不收——见 Step 3。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_dqn.py tests/test_evaluate.py`
Expected: `train` / `evaluate` not defined 等失败。

- [ ] **Step 3: 实现主循环 `train()` + 只测不训的 `evaluate`/`load_net`/`plot_curves`**

在 `dqn.py` 追加（全部收在本模块，避免 scripts 非包导致的 import 失败）：

```python
class _NoopTrainer:
    """基准赛专用占位：只提供 .train_step() 与 .buffer，no-op，确保只测不训。"""
    buffer = type("B", (), {"push": lambda *a, **k: None})()
    def train_step(self):
        return None


_NOOP = _NoopTrainer()


def _run_benchmark(candidate, opponent, n: int = 20) -> float:
    """candidate：fn(board)->action（贪心）；opponent：fn(board)->action；OK（黑）胜率。"""
    wins = 0
    for _ in range(n):
        board = Board(size=SIZE, win_len=WIN_LEN)
        info = play_episode(board, candidate, opponent, _NOOP)
        wins += int(info["winner"] == BLACK)
    return wins / n


def _default_evaluate(trainer) -> dict:
    free = lambda b: select_action(trainer.net, b, eps=0.0, device=trainer.device)
    return {
        "win_vs_random": _run_benchmark(free, random_opponent, 20),
        "win_vs_easy": _run_benchmark(free, make_minimax_opponent("easy"), 20),
    }


def evaluate(net: DQNNet, opponent, n: int = 20, device: str = "cpu") -> float:
    """只测不训：贪心（eps=0）打 n 局固定对手的胜率。"""
    free = lambda b: select_action(net, b, eps=0.0, device=device)
    return _run_benchmark(free, opponent, n)


def load_net(checkpoint: str, size: int = SIZE, channels: int = 64,
             device: str = "cpu") -> DQNNet:
    net = DQNNet(size=size, channels=channels)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()
    return net


def opening_policy_entropy(trainer) -> float:
    """空盘开局观测下的动作策略熵（越小越决定，读取网络中是否有局部结构）。"""
    with torch.no_grad():
        obs = torch.as_tensor(Board(size=SIZE, win_len=WIN_LEN).observation(),
                              dtype=torch.float32, device=trainer.device).unsqueeze(0)
        q = trainer.net(obs).squeeze(0).detach().cpu().numpy()
    return policy_entropy_from_logits(q, np.arange(SIZE * SIZE))


def train(trainer, episodes_per_era: int = 20_000, num_eras: int = 60,
          eps_start: float = 1.0, eps_end: float = 0.05,
          eps_decay: int = 720_000, seed: int = 0, device: str = "cpu",
          out_dir: str = "runs/e1", evaluate_fn=None, eval_n: int = 20
          ) -> list[dict]:
    """逐代自对弈主循环：B2 对手池、ε 退火、replay 累积、逐代基准赛写 metrics.jsonl。

    evaluate_fn 不传则用 _default_evaluate（vs random / vs minimax-easy 各 20 局）。
    """
    import json
    import pathlib
    out = pathlib.Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    opps = [random_opponent, make_minimax_opponent("easy")]   # B2：对半随机抽
    if evaluate_fn is None:
        evaluate_fn = _default_evaluate
    logs = []
    all_loss, all_steps = [], []
    for era in range(num_eras):
        for n_local in range(episodes_per_era):
            opp = opps[int(rng.random() >= 0.5)]
            eps = trainer.get_eps(era * episodes_per_era + n_local,
                                  eps_start, eps_end, eps_decay)
            board = Board(size=SIZE, win_len=WIN_LEN)
            candidate = lambda b: select_action(trainer.net, b, eps, device)
            info = play_episode(board, candidate, opp, trainer)
            all_steps.append(info["steps"])
            all_loss.extend(info["losses"])
        row = {
            "era": era,
            "steps": float(np.mean(all_steps[-episodes_per_era:])),
            "loss": float(np.mean(all_loss[-episodes_per_era:])) if all_loss else 0.0,
            "entropy": opening_policy_entropy(trainer),
            "eps": eps,
        }
        row.update(evaluate_fn(trainer))
        logs.append(row)
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        torch.save(trainer.net.state_dict(),
                   out / "checkpoints" / f"gen{era:04d}.pt")
    return logs


def plot_curves(metrics: list[dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eras = [m["era"] for m in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(eras, [m["win_vs_random"] for m in metrics])
    axes[0, 0].set_title("win vs random")
    axes[0, 1].plot(eras, [m["win_vs_easy"] for m in metrics])
    axes[0, 1].set_title("win vs minimax-easy")
    axes[1, 0].plot(eras, [m["steps"] for m in metrics])
    axes[1, 0].set_title("avg episode length")
    axes[1, 1].plot(eras, [m["loss"] for m in metrics])
    axes[1, 1].set_title("training loss")
    for ax in axes.flat:
        ax.set_xlabel("era")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
```

- [ ] **Step 4: 跑测试确认通过（端到端小冒烟 + 基准赛只测不训）**

Run: `python -m pytest tests/test_dqn.py tests/test_evaluate.py`
Expected: 全绿——`train` 产出 2 行 metrics、checkpoint 落盘、`evaluate` 胜率在界内。

- [ ] **Step 5: 写训练脚本 `scripts/train_dqn.py`（60 代驱动 + 画图）**

```python
"""课 01 · E1：完整配方 DQN 训练入口（headless，A2 快门派）。

用法：
    python scripts/train_dqn.py --epochs 60 --episodes-per-era 20000 \
        --batch-size 256 --device auto --exp-id e1

输出 runs/e1/：config.json + metrics.jsonl + curve.png + checkpoints/genNNNN.pt
"""

import argparse
import json
import pathlib

import torch

from gomoku.rl.dqn import DQNNet, DQNTrainer, train, plot_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 完整配方 DQN 训练")
    parser.add_argument("--epochs", type=int, default=60)          # 即 num_eras
    parser.add_argument("--episodes-per-era", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=720_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exp-id", default="e1")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path("runs") / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    net = DQNNet(size=15, channels=64)
    trainer = DQNTrainer(net, gamma=args.gamma, lr=args.lr,
                         batch_size=args.batch_size, target_sync_steps=10_000,
                         device=device)
    metrics = train(trainer, episodes_per_era=args.episodes_per_era,
                    num_eras=args.epochs, eps_start=args.eps_start,
                    eps_end=args.eps_end, eps_decay=args.eps_decay,
                    seed=args.seed, device=device, out_dir=str(out))
    plot_curves(metrics, str(out / "curve.png"))
    last = metrics[-1]
    print(f"done: {args.exp_id}  win_vs_easy={last['win_vs_easy']:.2f} "
          f"win_vs_random={last['win_vs_random']:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 写基准赛脚本 `scripts/evaluate_dqn.py`（加载 checkpoint 打固定对手）**

```python
"""课 01 · E1：基准赛——加载 checkpoint 打固定对手，衡量 RL 进步的固定标尺。

用法：
    python scripts/evaluate_dqn.py --checkpoint runs/e1/checkpoints/gen0059.pt \
        --opponent easy --n 100 --device auto
"""

import argparse

import torch

from gomoku.rl.dqn import evaluate, load_net, make_minimax_opponent, random_opponent


OPPONENTS = {
    "random": lambda: random_opponent,
    "easy": lambda: make_minimax_opponent("easy"),
    "medium": lambda: make_minimax_opponent("medium"),
    "hard": lambda: make_minimax_opponent("hard"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 基准赛")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", choices=list(OPPONENTS), default="easy")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_net(args.checkpoint, size=15, channels=64, device=device)
    opp = OPPONENTS[args.opponent]()
    rate = evaluate(net, opp, n=args.n, device=device)
    print(f"{args.opponent}: {rate:.2f} ({args.n} games)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 跑全量测试 + 构建**

Run: `python -m pytest`
Expected: 全部通过（既有 69 + 新增 E1 十几条），无回归。

- [ ] **Step 8: 写训练脚本并做一次 1 代 20 万局冒烟（可选，验证脚本真能跑通，不死循环/不 OOM）**

Run: `python scripts/train_dqn.py --epochs 1 --episodes-per-era 50 --batch-size 16 --device cpu --exp-id e1-smoke`
Expected: `runs/e1-smoke/` 生成 metrics.jsonl、curve.png、gen0000.pt，程序正常结束。

- [ ] **Step 9: Commit**

```bash
git add src/gomoku/rl/dqn.py scripts/train_dqn.py scripts/evaluate_dqn.py tests/test_dqn.py tests/test_evaluate.py
git commit -m "feat: E1 DQN 主循环 train() + 基准赛 evaluate + 训练/评估双脚本（60 代完整配方基线）"
```

> 本计划由 spec `docs/superpowers/specs/2026-09-06-e1-dqn-baseline-design.md` 约束，所有 dqn.py 文件由 Task 1-6 `Modify` 方式在同一文件上追加，各任务独立可验收、可回退。