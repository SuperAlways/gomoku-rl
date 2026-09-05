import numpy as np

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
DRAW = 3  # winner 取值：满盘平局

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    """15×15 无禁手五子棋棋盘：规则内核，唯一的状态权威。

    grid: int8 (15,15)，0 空 / 1 黑 / 2 白
    history: [(player, row, col), ...] 落子序
    winner: 0 进行中 / 1 黑 / 2 白 / 3 平局（终局时 current_player 停在胜者/最后一手方）
    """

    def __init__(self, size: int = BOARD_SIZE):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.current_player = BLACK
        self.winner = EMPTY
        self.history: list[tuple[int, int, int]] = []
        self.last_move: tuple[int, int] | None = None

    @property
    def game_over(self) -> bool:
        return self.winner != EMPTY

    def play(self, row: int, col: int) -> None:
        if self.game_over:
            raise ValueError("game is over")
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"out of board: ({row}, {col})")
        if self.grid[row, col] != EMPTY:
            raise ValueError(f"occupied: ({row}, {col})")
        player = self.current_player
        self.grid[row, col] = player
        self.history.append((player, row, col))
        self.last_move = (row, col)
        if self._has_five(row, col, player):
            self.winner = player
        elif len(self.history) == self.size * self.size:
            self.winner = DRAW
        else:
            self.current_player = 3 - player

    def undo(self) -> None:
        if not self.history:
            raise ValueError("no move to undo")
        player, row, col = self.history.pop()
        self.grid[row, col] = EMPTY
        self.winner = EMPTY
        self.current_player = player
        self.last_move = self.history[-1][1:] if self.history else None

    def valid_moves(self) -> np.ndarray:
        return self.grid == EMPTY

    def move_to_action(self, row: int, col: int) -> int:
        return row * self.size + col

    def action_to_move(self, action: int) -> tuple[int, int]:
        return divmod(action, self.size)

    def observation(self, player: int = BLACK) -> np.ndarray:
        """(2,15,15) 特征平面：player 的棋子 / 对手的棋子（M2 起 NN 输入）"""
        return np.stack([
            (self.grid == player).astype(np.float32),
            (self.grid == 3 - player).astype(np.float32),
        ])

    def _has_five(self, row: int, col: int, player: int) -> bool:
        # 只以最后一手为中心查 4 方向，不做全盘扫描
        for dr, dc in _DIRS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 5:
                return True
        return False

    def winning_line(self) -> list[tuple[int, int]] | None:
        """获胜五连坐标（含最后一手的一段 5 格）；未终局或平局返回 None"""
        if not self.game_over or self.winner == DRAW:
            return None
        row, col = self.last_move
        player = self.winner
        for dr, dc in _DIRS:
            cells = [(row, col)]
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.size and 0 <= c < self.size \
                        and self.grid[r, c] == player:
                    cells.append((r, c))
                    r += sign * dr
                    c += sign * dc
            if len(cells) >= 5:
                return cells[:5]
        return None
