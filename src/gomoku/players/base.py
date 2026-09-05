from abc import ABC, abstractmethod

from gomoku.core import Board


class Player(ABC):
    """所有对弈者的统一抽象：给定棋盘，返回 action 整数（row*15+col）。

    这是对战平台与未来所有 RL 算法的接入点——NNPlayer 加载任意
    权重后同样实现此接口。
    """

    name: str = "player"

    @abstractmethod
    def select_move(self, board: Board) -> int: ...
