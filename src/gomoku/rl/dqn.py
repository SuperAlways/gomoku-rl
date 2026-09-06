"""课 01 · E1：15×15 完整配方 DQN。

三件套（DQNNet / ReplayBuffer / DQNTrainer）+ 自对弈收集 play_episode，
全部不依赖 Arena（带 DB 太重）；训练循环手攥 Board，与 tabular.py 同风格。
"""

import numpy as np
import torch
from torch import nn

from gomoku.core import BLACK, DRAW, WHITE, Board

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
    illegal = np.array([i for i in range(SIZE * SIZE) if i not in legal.tolist()],
                       dtype=np.int64)
    if len(illegal) > 0:
        q[illegal] = -np.inf
    return int(legal[q[legal].argmax()])


class DQNTrainer:
    """DQN 训练核心：replay 采样 → TD 靶子（target 网络 + 合法 mask）→ MSE → 一步 Adam。

    buffer 默认 50 万条；target 每 target_sync_steps 次 train_step 同步一次。
    """

    def __init__(self, net: DQNNet, gamma: float = 0.99, lr: float = 1e-4,
                 batch_size: int = 256, target_sync_steps: int = 10_000,
                 learn_every: int = 4, device: str = "cpu"):
        self.net = net.to(device)
        self.target_net = DQNNet(net.size, net.features[0].out_channels).to(device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_steps = target_sync_steps
        self.learn_every = learn_every
        self._learn_counter = 0
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
        self._learn_counter += 1
        if self._learn_counter % self.learn_every != 0:
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
        sync_every = max(1, self.target_sync_steps // self.learn_every)
        if self.steps_done % sync_every == 0:
            self.sync_target()
        return float(loss.item())

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
    用当时最新 observation 作 next_state。终局回填稀疏 +1/-1/0。
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


class _NoopTrainer:
    """基准赛专用占位：只提供 .train_step() 与 .buffer，no-op，确保只测不训。"""
    buffer = type("B", (), {"push": lambda *a, **k: None})()
    def train_step(self):
        return None


_NOOP = _NoopTrainer()


def _run_benchmark(candidate, opponent, n: int = 20) -> float:
    """candidate：fn(board)->action（贪心）；opponent：fn(board)->action；黑方胜率。"""
    wins = 0
    for _ in range(n):
        board = Board(size=SIZE, win_len=WIN_LEN)
        info = play_episode(board, candidate, opponent, _NOOP)
        wins += int(info["winner"] == BLACK)
    return wins / n


def _default_evaluate(trainer, n: int = 20) -> dict:
    free = lambda b: select_action(trainer.net, b, eps=0.0, device=trainer.device)
    return {
        "win_vs_random": _run_benchmark(free, random_opponent, n),
        "win_vs_easy": _run_benchmark(free, make_minimax_opponent("easy"), n),
    }


def evaluate(net: DQNNet, opponent, n: int = 20, device: str = "cpu") -> float:
    """只测不训：贪心（eps=0）打 n 局固定对手的胜率。"""
    net.eval()
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
    np.random.seed(seed)
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
            ep_loss = float(np.mean(info["losses"])) if info["losses"] else 0.0
            all_loss.append(ep_loss)
        row = {
            "era": era,
            "steps": float(np.mean(all_steps[-episodes_per_era:])),
            "loss": float(np.mean(all_loss[-episodes_per_era:])) if all_loss else 0.0,
            "entropy": opening_policy_entropy(trainer),
            "eps": eps,
        }
        row.update(evaluate_fn(trainer, eval_n))
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