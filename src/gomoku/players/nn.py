from gomoku.core import Board
from gomoku.players.base import Player


class NNPlayer(Player):
    """NN 权重 agent 骨架。M2（DQN 课）实现 select_move：

    加载 model_path 权重 → board.observation(player) → 前向
    → mask 无效点（valid_moves）→ 返回 action（row*15+col）。
    """

    name = "nn"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def select_move(self, board: Board) -> int:
        raise NotImplementedError("M2 实现")