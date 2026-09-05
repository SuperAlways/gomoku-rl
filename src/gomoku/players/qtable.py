"""QTablePlayer：加载表格 Q-learning 产出的 qtable.json 参与对战。"""

import random

import numpy as np

from gomoku.players.base import Player
from gomoku.rl.tabular import load_qtable, state_key


class QTablePlayer(Player):
    """查表 + 合法动作 mask + argmax；查不到的局面 → 随机合法位。"""

    def __init__(self, qtable_path: str, name: str = "qtable-3x3"):
        self.name = name
        self.q, self.meta = load_qtable(qtable_path)

    def select_move(self, board) -> int:
        legal = np.flatnonzero(board.valid_moves())
        qs = self.q.get(state_key(board))
        if qs is None:
            return int(random.choice(legal.tolist()))
        return int(legal[np.argmax(qs[legal])])