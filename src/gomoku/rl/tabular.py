"""表格 Q-learning（3×3 连三制，课 01 E0）。

状态键 = grid.tobytes() + bytes([current_player])：同一局面按行棋方区分。
Q 表 = dict[bytes, np.ndarray(size*size)]，初始全 0；合法动作按棋盘实况 mask。
"""

import base64
import json

import numpy as np

from gomoku.core import Board, BLACK, WHITE, DRAW, _DIRS


SIZE = 3
WIN_LEN = 3


def state_key(board: Board) -> bytes:
    return board.grid.tobytes() + bytes([board.current_player])


def _legal_actions(board: Board) -> np.ndarray:
    return np.flatnonzero(board.valid_moves())


def select_action(q: dict, board: Board, eps: float,
                  rng: np.random.Generator) -> int:
    """ε-greedy：以 eps 随机走合法位，否则查表 argmax（查不到的局面→随机）。"""
    legal = _legal_actions(board)
    if rng.random() < eps:
        return int(rng.choice(legal))
    key = state_key(board)
    if key not in q:
        return int(rng.choice(legal))
    return int(legal[np.argmax(q[key][legal])])


def random_opponent(board: Board, rng: np.random.Generator) -> int:
    return int(rng.choice(_legal_actions(board)))


def _completes_line(board: Board, row: int, col: int, player: int) -> bool:
    """假设 player 落 (row,col) 是否成 win_len 连（不改动棋盘）。"""
    for dr, dc in _DIRS:
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < board.size and 0 <= c < board.size \
                    and board.grid[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= board.win_len:
            return True
    return False


def blocker_opponent(board: Board, rng: np.random.Generator) -> int:
    """贪心脚本：对方（学员）两连则必堵，否则随机。固定不可学。"""
    legal = _legal_actions(board)
    threat = 3 - board.current_player
    for a in legal:
        r, c = board.action_to_move(int(a))
        if _completes_line(board, r, c, threat):
            return int(a)
    return int(rng.choice(legal))


OPPONENTS = {"random": random_opponent, "blocker": blocker_opponent}


def td_backfill(q: dict, trajectory: list[tuple[bytes, int]], reward: float,
                gamma: float, lr: float) -> list[float]:
    """终局奖励反向逐手回填学员的着；返回各手 TD error（更新前）。

    最后一手 done=1：target = reward；其余 target = γ·max_{合法 a'} Q[s'][a']。
    反向遍历时 Q[s'][a'] 恰是上一轮刚更新过的下一决策状态。
    合法性从状态键解码棋盘得出——从未更新过的非法位（Q=0）不许污染 bootstrap。
    """
    errs = []
    target = reward
    for key, action in reversed(trajectory):
        errs.append(abs(target - q[key][action]))
        q[key][action] += lr * (target - q[key][action])
        grid = np.frombuffer(key[:-1], dtype=np.int8)
        legal = np.flatnonzero(grid == 0)
        target = gamma * float(np.max(q[key][legal]))
    return errs


def policy_entropy(q_values: np.ndarray, legal: np.ndarray,
                   tau: float = 0.3) -> float:
    """Q 表导出的策略熵：softmax(Q/τ) 在合法动作上的 Shannon 熵。"""
    z = q_values[legal] / tau
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def run_episode(q: dict, opponent, eps: float, rng: np.random.Generator,
                gamma: float, lr: float,
                size: int = SIZE, win_len: int = WIN_LEN) -> dict:
    """学员固定执黑先行打一局，终局后回填；返回单局指标。"""
    board = Board(size=size, win_len=win_len)
    trajectory: list[tuple[bytes, int]] = []
    entropies: list[float] = []
    while not board.game_over:
        if board.current_player == BLACK:
            key = state_key(board)
            q.setdefault(key, np.zeros(size * size))
            action = select_action(q, board, eps, rng)
            trajectory.append((key, action))
            entropies.append(policy_entropy(q[key], _legal_actions(board)))
        else:
            action = opponent(board, rng)
        row, col = board.action_to_move(action)
        board.play(row, col)
    reward = {BLACK: 1.0, WHITE: -1.0, DRAW: 0.0}[board.winner]
    td_errors = td_backfill(q, trajectory, reward, gamma, lr)
    return {
        "winner": int(board.winner),
        "steps": len(board.history),
        "td_error": float(np.mean(td_errors)),
        "entropy": float(np.mean(entropies)),
    }


def train(opponent: str, episodes: int,
          eps_start: float = 1.0, eps_end: float = 0.05,
          eps_decay_episodes: int = 3000,
          gamma: float = 0.99, lr: float = 0.5,
          seed: int | None = None,
          size: int = SIZE, win_len: int = WIN_LEN
          ) -> tuple[dict[bytes, np.ndarray], list[dict]]:
    """ε 线性退火跑 episodes 局；返回 (Q 表, 逐局 metrics 行)。"""
    rng = np.random.default_rng(seed)
    opp = OPPONENTS[opponent]
    q: dict[bytes, np.ndarray] = {}
    rows: list[dict] = []
    for episode in range(episodes):
        frac = min(1.0, episode / max(1, eps_decay_episodes))
        eps = eps_start + (eps_end - eps_start) * frac
        info = run_episode(q, opp, eps, rng, gamma, lr, size, win_len)
        rows.append({"episode": episode, **info, "eps": eps})
    return q, rows


def save_qtable(q: dict, path: str, meta: dict) -> None:
    payload = {
        "meta": meta,
        "states": {base64.b64encode(k).decode("ascii"): v.tolist()
                   for k, v in q.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_qtable(path: str) -> tuple[dict[bytes, np.ndarray], dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    q = {base64.b64decode(k): np.asarray(v, dtype=np.float64)
         for k, v in payload["states"].items()}
    return q, payload["meta"]


def sliding_win_rate(rows: list[dict], window: int = 100) -> list[float]:
    """学员（黑）胜率的滑动平均；窗口不足时按已有局数平均。"""
    out = []
    for i in range(len(rows)):
        chunk = rows[max(0, i - window + 1): i + 1]
        out.append(sum(r["winner"] == BLACK for r in chunk) / len(chunk))
    return out
