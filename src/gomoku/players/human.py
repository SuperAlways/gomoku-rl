from gomoku.core import Board
from gomoku.players.base import Player


class HumanPlayer(Player):
    """真人。M0 中不自主思考——落子来自 Web UI 的请求。"""

    def __init__(self, name: str = "human"):
        self.name = name

    def select_move(self, board: Board) -> int:
        raise NotImplementedError("human moves come from the UI")
