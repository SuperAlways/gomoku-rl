"""Minimax 专家系统。

评分体系翻译自 references/gobang-js/src/ai/（shape.js/eval.js 教程）：
全盘按"线"扫描成模式串做棋型匹配。本文件上半部是评估函数，
下半部（Task 5）是 α-β 搜索。
"""
import numpy as np

from gomoku.core import Board
from gomoku.players.base import Player

# 棋型打分表：从上到下优先级递减，同一方向/同一条线匹配到即停（只计最高棋型）
# 模式串：'1'=本方棋子, '2'=对方棋子, '0'=空位
PATTERNS = (
    (10_000_000, ("11111",)),                                   # 连五
    (1_000_000, ("011110",)),                                   # 活四
    (100_000, ("211110", "011112", "10111", "11011", "11101")), # 冲四
    (10_000, ("01110", "010110", "011010")),                    # 活三
    (1_000, ("211100", "001112", "211010", "010112",
             "210110", "011012", "10011", "11001", "10101")),   # 眠三
    (500, ("001100", "010100", "001010")),                      # 活二
    (100, ("211000", "000112", "210100", "001012",
           "210010", "010012", "10001")),                       # 眠二
)

_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
_SWAP = str.maketrans("12", "21")


def _view(line: str, player: int) -> str:
    """把 '0'/'1'/'2' 棋盘串转成"本方=1 对方=2"的视角串"""
    return line if player == 1 else line.translate(_SWAP)


def _score_line(line: str) -> int:
    for score, pats in PATTERNS:
        for p in pats:
            if p in line:
                return score * line.count(p)
    return 0


def _all_lines(board: Board) -> list[str]:
    """棋盘全部线：15 行 + 15 列 + 两组对角线（长度 ≥5），值为 '0'/'1'/'2'"""
    g, n = board.grid, board.size
    lines = ["".join(map(str, g[r])) for r in range(n)]
    lines += ["".join(map(str, g[:, c])) for c in range(n)]
    for k in range(-(n - 5), n - 4):
        lines.append("".join(map(str, np.diag(g, k))))
        lines.append("".join(map(str, np.diag(np.fliplr(g), k))))
    return lines


def _side_score(lines: list[str], player: int) -> int:
    total = 0
    ch = str(player)
    for line in lines:
        if ch in line:
            total += _score_line(_view(line, player))
    return total


def evaluate(board: Board, player: int) -> int:
    """全盘评估：player 视角 = 我方棋型分 − 2×对方棋型分（防守加权）

    评估非零和（防守加权 2×），negamax 逐层取负是近似，浅搜索（depth≤4）下实践可行。
    """
    lines = _all_lines(board)
    return _side_score(lines, player) - 2 * _side_score(lines, 3 - player)


def _window_score(board: Board, r: int, c: int, player: int) -> int:
    """假设 player 落子 (r,c)，4 方向 9 格窗口内的最高棋型分之和"""
    g, n = board.grid, board.size
    total = 0
    for dr, dc in _DIRS:
        cells = []
        for i in range(-4, 5):
            rr, cc = r + i * dr, c + i * dc
            if 0 <= rr < n and 0 <= cc < n:
                v = int(g[rr, cc])
                if (rr, cc) == (r, c):
                    cells.append("1")
                else:
                    cells.append("0" if v == 0 else ("1" if v == player else "2"))
        s = "".join(cells)
        for score, pats in PATTERNS:
            if any(p in s for p in pats):
                total += score
                break
    return total


def point_score(board: Board, r: int, c: int, player: int) -> int:
    """落点启发分 = 我在此点的进攻分 + 对方在此点的价值（防守分）"""
    return _window_score(board, r, c, player) + _window_score(board, r, c, 3 - player)


def raw_candidates(board: Board) -> list[tuple[int, int]]:
    """已有棋子切比雪夫距离 2 内的空点；空盘返回天元"""
    g, n = board.grid, board.size
    stones = np.argwhere(g != 0)
    if len(stones) == 0:
        return [(n // 2, n // 2)]
    cand = set()
    for r, c in stones:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < n and 0 <= cc < n and g[rr, cc] == 0:
                    cand.add((rr, cc))
    return sorted(cand)


def candidates(board: Board, limit: int = 12) -> list[tuple[int, int]]:
    """候选点 = raw_candidates 按 point_score 降序取前 limit"""
    me = board.current_player
    scored = sorted(
        raw_candidates(board),
        key=lambda rc: point_score(board, rc[0], rc[1], me),
        reverse=True,
    )
    return scored[:limit]


WIN = 10**9

LEVELS = {"easy": 1, "medium": 2, "hard": 4}


def _immediate(board: Board) -> tuple[int, int] | None:
    """hard 顶层快检：我能立即连五就下；对方能立即连五就堵。"""
    me = board.current_player
    for who in (me, 3 - me):
        for r, c in raw_candidates(board):
            board.play(r, c)
            won = board.winner == who
            board.undo()
            if won:
                return (r, c)
    return None


def _search(board: Board, depth: int, alpha: int, beta: int) -> int:
    """负极大 α-β：返回值是"当前行棋方"视角的分。

    终局时 current_player 停在刚落子的一方（Board.play 的约定），
    所以 winner == current_player 即"当前行棋方刚赢"。
    """
    if board.game_over:
        if board.winner == 3:
            return 0
        return WIN if board.winner == board.current_player else -WIN
    if depth == 0:
        return evaluate(board, board.current_player)
    best = -10**9
    for r, c in candidates(board, limit=12):
        board.play(r, c)
        score = -_search(board, depth - 1, -beta, -alpha)
        board.undo()
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


class MinimaxPlayer(Player):
    """α-β 剪枝 Minimax，难度 = 搜索深度（easy/medium/hard = 1/2/4 层）。"""

    def __init__(self, level: str = "medium"):
        if level not in LEVELS:
            raise ValueError(f"unknown level: {level}")
        self.level = level
        self.depth = LEVELS[level]
        self.name = f"minimax-{level}"

    def select_move(self, board: Board) -> int:
        if board.game_over:
            raise ValueError("game is over")
        if not board.history:
            center = board.size // 2
            return center * board.size + center
        if self.level == "hard":
            forced = _immediate(board)
            if forced:
                return forced[0] * board.size + forced[1]
        best_action, best_score = None, -10**9
        for r, c in candidates(board, limit=12):
            board.play(r, c)
            score = -_search(board, self.depth - 1, -10**9, 10**9)
            board.undo()
            if score > best_score:
                best_action, best_score = (r, c), score
        if best_action is None:
            # 候选点耗尽（极端满盘），退回任一合法步
            legal = np.flatnonzero(board.valid_moves())
            return int(legal[0])
        return best_action[0] * board.size + best_action[1]