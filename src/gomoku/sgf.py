"""SGF 棋谱导出。坐标映射：(row, col) → 两字母 a–o，(7,7)→"hh"。"""

_LETTER = {1: "B", 2: "W"}
_RE = {"black_win": "B+", "white_win": "W+", "draw": "0"}


def _coord(row: int, col: int) -> str:
    return chr(ord("a") + row) + chr(ord("a") + col)


def build_sgf(black_name: str, white_name: str, result: str,
              moves: list[tuple[int, int, int]]) -> str:
    """moves: [(player, row, col), ...]（Board.history / moves 表同构）。"""
    parts = [f"(;FF[4]SZ[15]PB[{black_name}]PW[{white_name}]"]
    if result in _RE:
        parts.append(f"RE[{_RE[result]}]")
    parts.append("\n")
    for player, row, col in moves:
        parts.append(f";{_LETTER[player]}[{_coord(row, col)}]")
    parts.append(")")
    return "".join(parts)