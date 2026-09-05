# M0 实施计划：五子棋环境 + Minimax 专家系统 + Web UI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付 M0 三件套——可单测的 15×15 无禁手五子棋规则内核、三档难度的 Minimax 专家系统、人/机任意组合对战的 Web UI（FastAPI + React），棋谱入 SQLite。

**Architecture:** Board 是唯一状态权威（纯 Python + numpy）；Player 抽象 `select_move(board) -> int`，MinimaxPlayer 是第一个实现；Arena 是 AI-vs-AI 裁判并负责写库；FastAPI 会话驱动 Board（人走 + AI 应手）；React 前端纯视图。评估函数用"全盘线扫描 + 棋型模式串匹配"（翻译自 references/gobang-js/src/ai/）。

**Tech Stack:** Python ≥3.11、numpy、FastAPI、uvicorn、sqlite3（标准库）、pytest、httpx（TestClient）；Vite 5 + React 18。

**Spec:** `docs/superpowers/specs/2026-09-05-m0-env-design.md`（本计划全部需求出自它）

## Global Constraints

- 棋盘固定 15×15，无禁手自由规则；格值 0 空 / 1 黑 / 2 白；`winner`: 0 进行中 / 1 黑 / 2 白 / 3 平局
- action 整数编码 `row * 15 + col`，全项目统一
- 统一 Player 抽象：`select_move(self, board: Board) -> int`，对手名 `human` / `minimax-easy|medium|hard`
- 评分表数值固定：连五 10_000_000、活四 1_000_000、冲四 100_000、活三 10_000、眠三 1_000、活二 500、眠二 100；叶子评估 = 我方分 − 2×对方分
- 难度 = 深度分档：easy/medium/hard = depth 1/2/4，hard 另有顶层"立即取胜/必挡"快检；M0 不做思考限时
- SQLite 单文件 `gomoku.db`（gitignore），只有 `games`/`moves` 两张表；RL 训练数据不进库
- 依赖只允许：numpy、fastapi、uvicorn、pydantic、pytest、httpx（dev）
- Windows + bash：路径用正斜杠；`sqlite3` CLI 不可用，查库用 `python -c`
- 不得修改 `references/` 下任何文件；每个任务结束必须 commit；全部完成后 push

---

### Task 1: 项目骨架与开发环境

**Files:**
- Create: `pyproject.toml`
- Create: `src/gomoku/__init__.py`
- Create: `tests/test_smoke.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: 可 `pip install -e` 的包 `gomoku`；pytest 可运行；后续所有任务 import `gomoku.*`

- [ ] **Step 1: 写 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gomoku-rl"
version = "0.1.0"
description = "以五子棋为载体的强化学习实战项目"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2",
]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 写包入口与冒烟测试**

`src/gomoku/__init__.py`：

```python
__version__ = "0.1.0"
```

`tests/test_smoke.py`：

```python
def test_import():
    import gomoku
    assert gomoku.__version__ == "0.1.0"
```

- [ ] **Step 3: 更新 .gitignore（追加）**

```
__pycache__/
*.egg-info/
gomoku.db
frontend/node_modules/
frontend/dist/
```

- [ ] **Step 4: 安装并验证**

Run: `python --version && pip install -e ".[dev]" && pytest -q`
Expected: Python ≥3.11；安装成功；`1 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/gomoku/__init__.py tests/test_smoke.py .gitignore
git commit -m "feat: 项目骨架（src 布局 + pytest）"
```

---

### Task 2: Board 规则内核（core.py）

**Files:**
- Create: `src/gomoku/core.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: 无（仅 numpy）
- Produces: `Board(size=15)`；`Board.play(row, col)`（非法则 `ValueError`）；`Board.undo()`；`Board.valid_moves() -> np.ndarray(bool)`；`Board.move_to_action(row,col)/action_to_move(action)`；`Board.observation(player) -> np.float32 (2,15,15)`；属性 `grid/current_player/winner/last_move/history/game_over`；常量 `EMPTY=0, BLACK=1, WHITE=2, DRAW=3, BOARD_SIZE=15`。后续所有任务依赖这些名字。

- [ ] **Step 1: 写失败测试 tests/test_core.py**

```python
import numpy as np
import pytest

from gomoku.core import Board, BLACK, WHITE, DRAW, BOARD_SIZE


def test_initial_state():
    b = Board()
    assert b.size == BOARD_SIZE == 15
    assert b.current_player == BLACK
    assert not b.game_over
    assert b.winner == 0
    assert b.last_move is None
    assert b.history == []
    assert b.valid_moves().sum() == 225


def test_horizontal_win():
    b = Board()
    for r, c in [(7, 4), (8, 0), (7, 5), (8, 1), (7, 6), (8, 2), (7, 7), (8, 3)]:
        b.play(r, c)
    assert not b.game_over
    b.play(7, 8)
    assert b.winner == BLACK and b.game_over


def test_vertical_win():
    b = Board()
    for r, c in [(3, 7), (3, 8), (4, 7), (4, 8), (5, 7), (5, 8), (6, 7), (6, 8)]:
        b.play(r, c)
    b.play(7, 7)
    assert b.winner == BLACK


def test_diagonal_win():
    b = Board()
    for r, c in [(5, 5), (0, 0), (6, 6), (0, 1), (7, 7), (0, 2), (8, 8), (0, 3)]:
        b.play(r, c)
    b.play(9, 9)
    assert b.winner == BLACK


def test_anti_diagonal_win():
    b = Board()
    for r, c in [(5, 9), (0, 14), (6, 8), (0, 13), (7, 7), (0, 12), (8, 6), (0, 11)]:
        b.play(r, c)
    b.play(9, 5)
    assert b.winner == BLACK


def test_draw_full_board():
    # 颜色 = (r//2 + c) % 2：行内相邻异色、列内同色最多连 2、
    # 两个对角方向同色最多连 2 —— 全盘不存在任何方向五连。
    # 黑格恰 113 个 = 黑方手数（先手 113 手），白格 112 个。
    b = Board()
    black_cells = [(r, c) for r in range(15) for c in range(15) if (r // 2 + c) % 2 == 0]
    white_cells = [(r, c) for r in range(15) for c in range(15) if (r // 2 + c) % 2 == 1]
    assert len(black_cells) == 113 and len(white_cells) == 112
    for r, c in black_cells + white_cells:  # play() 自动交替黑白
        b.play(r, c)
    assert b.winner == DRAW and len(b.history) == 225


def test_undo_restores_state():
    b = Board()
    b.play(7, 7)
    b.play(7, 8)
    b.undo()
    assert b.grid[7, 8] == 0
    assert b.current_player == WHITE
    assert b.last_move == (7, 7)
    assert len(b.history) == 1
    b.undo()
    assert b.last_move is None and b.current_player == BLACK


def test_undo_after_win():
    b = Board()
    for r, c in [(7, 4), (8, 0), (7, 5), (8, 1), (7, 6), (8, 2), (7, 7), (8, 3), (7, 8)]:
        b.play(r, c)
    assert b.game_over
    b.undo()
    assert not b.game_over and b.winner == 0


def test_action_roundtrip():
    b = Board()
    for a in range(225):
        r, c = b.action_to_move(a)
        assert b.move_to_action(r, c) == a
    assert b.action_to_move(113) == (7, 8)


def test_observation_planes():
    b = Board()
    b.play(7, 7)
    b.play(7, 8)
    obs = b.observation(BLACK)
    assert obs.shape == (2, 15, 15) and obs.dtype == np.float32
    assert obs[0][7, 7] == 1.0 and obs[0][7, 8] == 0.0
    assert obs[1][7, 8] == 1.0


def test_invalid_moves():
    b = Board()
    b.play(7, 7)
    with pytest.raises(ValueError):
        b.play(7, 7)      # 已占用
    with pytest.raises(ValueError):
        b.play(15, 0)     # 越界
    with pytest.raises(ValueError):
        b.play(-1, 0)     # 越界
    b2 = Board()
    for r, c in [(7, 4), (8, 0), (7, 5), (8, 1), (7, 6), (8, 2), (7, 7), (8, 3), (7, 8)]:
        b2.play(r, c)
    with pytest.raises(ValueError):
        b2.play(0, 0)     # 终局后
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_core.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'gomoku.core'`）

- [ ] **Step 3: 写实现 src/gomoku/core.py**

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_core.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/gomoku/core.py tests/test_core.py
git commit -m "feat: Board 规则内核（落子/胜负/undo/observation/action 编码）"
```

---

### Task 3: SQLite 存储（db.py）

**Files:**
- Create: `src/gomoku/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: 无
- Produces: `connect(path="gomoku.db") -> sqlite3.Connection`（已建表，`row_factory=sqlite3.Row`）；`create_game(conn, black_name, white_name) -> int`；`record_move(conn, game_id, move)`，`move=(player,row,col)`；`finish_game(conn, game_id, result, total_moves)`；`list_games(conn, limit=50) -> list[dict]`。Task 6/7 依赖。

- [ ] **Step 1: 写失败测试 tests/test_db.py**

```python
from gomoku.db import connect, create_game, record_move, finish_game, list_games


def test_game_lifecycle(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    gid = create_game(conn, "human", "minimax-hard")
    assert isinstance(gid, int)
    record_move(conn, gid, (1, 7, 7))
    record_move(conn, gid, (2, 7, 8))
    finish_game(conn, gid, "black_win", 2)
    games = list_games(conn)
    assert len(games) == 1
    g = games[0]
    assert g["black_name"] == "human" and g["white_name"] == "minimax-hard"
    assert g["result"] == "black_win" and g["total_moves"] == 2
    (n,) = conn.execute("SELECT COUNT(*) FROM moves WHERE game_id=?", (gid,)).fetchone()
    assert n == 2


def test_ongoing_result_default(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    gid = create_game(conn, "a", "b")
    (result,) = conn.execute("SELECT result FROM games WHERE id=?", (gid,)).fetchone()
    assert result == "ongoing"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_db.py -q`
Expected: FAIL（`No module named 'gomoku.db'`）

- [ ] **Step 3: 写实现 src/gomoku/db.py**

```python
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    black_name TEXT NOT NULL,
    white_name TEXT NOT NULL,
    result TEXT NOT NULL DEFAULT 'ongoing',
    total_moves INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS moves (
    game_id INTEGER NOT NULL REFERENCES games(id),
    seq INTEGER NOT NULL,
    player INTEGER NOT NULL,
    row INTEGER NOT NULL,
    col INTEGER NOT NULL,
    PRIMARY KEY (game_id, seq)
);
"""


def connect(path: str = "gomoku.db") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def create_game(conn: sqlite3.Connection, black_name: str, white_name: str) -> int:
    cur = conn.execute(
        "INSERT INTO games (black_name, white_name) VALUES (?, ?)",
        (black_name, white_name),
    )
    conn.commit()
    return cur.lastrowid


def record_move(conn: sqlite3.Connection, game_id: int, move: tuple[int, int, int]) -> None:
    player, row, col = move
    (seq,) = conn.execute(
        "SELECT COUNT(*) FROM moves WHERE game_id = ?", (game_id,)
    ).fetchone()
    conn.execute(
        "INSERT INTO moves (game_id, seq, player, row, col) VALUES (?, ?, ?, ?, ?)",
        (game_id, seq + 1, player, row, col),
    )
    conn.commit()


def finish_game(conn: sqlite3.Connection, game_id: int, result: str, total_moves: int) -> None:
    conn.execute(
        "UPDATE games SET result = ?, total_moves = ? WHERE id = ?",
        (result, total_moves, game_id),
    )
    conn.commit()


def list_games(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM games ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_db.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/gomoku/db.py tests/test_db.py
git commit -m "feat: SQLite 存储（games/moves 表 + 生命周期函数）"
```

---

### Task 4: 棋型评估函数（minimax.py 上半）

**Files:**
- Create: `src/gomoku/players/__init__.py`（空文件）
- Create: `src/gomoku/players/minimax.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: Task 2 的 `Board/BLACK/WHITE`
- Produces（同文件，Task 5 续写；本任务只写评估半部）：
  - `PATTERNS`: tuple[(score, tuple[str, ...]), ...] —— 模式串 `'1'=本方 '2'=对方 '0'=空`
  - `evaluate(board, player) -> int`：全盘分，player 视角 = 我方 − 2×对方
  - `point_score(board, row, col, player) -> int`：假设落子的进攻+防守启发分
  - `raw_candidates(board) -> list[(r, c)]`：棋子切比雪夫距离 2 内的空点
  - `candidates(board, limit=12) -> list[(r, c)]`：按 point_score 降序取前 limit

- [ ] **Step 1: 写失败测试 tests/test_evaluation.py**

```python
from gomoku.core import Board, BLACK
from gomoku.players.minimax import evaluate, point_score, raw_candidates, candidates


def _board_from(moves):
    b = Board()
    for r, c in moves:
        b.play(r, c)
    return b


def test_empty_board_eval_zero():
    assert evaluate(Board(), BLACK) == 0


def test_live_three_detected():
    # 黑活三 (7,6)(7,7)(7,8)；白子三颗分散，不构成棋型
    b = _board_from([(7, 6), (1, 1), (7, 7), (3, 3), (7, 8), (5, 5)])
    score_black = evaluate(b, BLACK)
    assert score_black >= 10_000          # 至少一个活三
    assert score_black > evaluate(b, 2)   # 黑方视角优于白方


def test_single_stones_score_zero():
    # 全是孤立单子时不该有棋型分
    b = _board_from([(1, 1), (3, 3), (5, 5), (7, 7), (9, 9), (11, 11)])
    assert evaluate(b, 1) == 0 and evaluate(b, 2) == 0
```

注意：`(1,1)(3,3)(5,5)(7,7)(9,9)` 分布在主对角线上但间隔 2，无五连也无棋型；若此断言失败说明模式表误报，需要修表而不是修测试。

```python
def test_point_score_ranks_extension():
    b = _board_from([(7, 6), (1, 1), (7, 7), (3, 3), (7, 8), (5, 5)])
    assert point_score(b, 7, 5, BLACK) >= 10_000     # 活三延伸点
    assert point_score(b, 7, 9, BLACK) >= 10_000     # 另一端
    assert point_score(b, 12, 12, BLACK) < 1_000     # 无关点


def test_raw_candidates_near_stones():
    b = _board_from([(7, 7), (3, 3)])
    cands = raw_candidates(b)
    assert all(abs(r - sr) <= 2 and abs(c - sc) <= 2
               for r, c in cands for sr, sc in [(7, 7), (3, 3)]
               if abs(r - sr) <= 2 and abs(c - sc) <= 2)
    assert len(cands) < 225 and len(cands) > 20


def test_candidates_sorted_and_capped():
    b = _board_from([(7, 7), (7, 8), (3, 3), (4, 4)])
    cands = candidates(b, limit=12)
    assert len(cands) == 12
    assert all(b.grid[r, c] == 0 for r, c in cands)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_evaluation.py -q`
Expected: FAIL（`No module named 'gomoku.players.minimax'`）

- [ ] **Step 3: 写评估半部 src/gomoku/players/minimax.py**

```python
"""Minimax 专家系统。

评分体系翻译自 references/gobang-js/src/ai/（shape.js/eval.js 教程）：
全盘按"线"扫描成模式串做棋型匹配。本文件上半部是评估函数，
下半部（Task 5）是 α-β 搜索。
"""
import numpy as np

from gomoku.core import Board

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
    total = 0
    for score, pats in PATTERNS:
        for p in pats:
            if p in line:
                total += score * line.count(p)
                break
    return total


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
    """全盘评估：player 视角 = 我方棋型分 − 2×对方棋型分（防守加权）"""
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_evaluation.py -q`
Expected: `5 passed`（若 `test_single_stones_score_zero` 失败：模式表误报，检查 PATTERNS 中混入了过短/过于宽松的模式，修表）

- [ ] **Step 5: Commit**

```bash
git add src/gomoku/players/__init__.py src/gomoku/players/minimax.py tests/test_evaluation.py
git commit -m "feat: 棋型评估（线扫描+模式匹配，翻译 gobang-js 评分体系）"
```

---

### Task 5: Player 基类 + α-β 搜索与三档难度（minimax.py 下半）

**Files:**
- Create: `src/gomoku/players/base.py`
- Create: `src/gomoku/players/human.py`
- Modify: `src/gomoku/players/minimax.py`（追加搜索部分）
- Test: `tests/test_minimax.py`

**Interfaces:**
- Consumes: Task 2 `Board`；Task 4 `candidates/evaluate/raw_candidates`
- Produces:
  - `Player(ABC)`：属性 `name: str`；抽象方法 `select_move(self, board: Board) -> int`（action 整数）
  - `HumanPlayer(name="human")`：`select_move` 抛 `NotImplementedError`（真人落子来自 UI）
  - `MinimaxPlayer(level="medium")`：`name` 自动为 `minimax-{level}`；`select_move` 返回 action
  - 模块级 `WIN = 10**9`

- [ ] **Step 1: 写失败测试 tests/test_minimax.py**

```python
import pytest

from gomoku.core import Board
from gomoku.players.base import Player
from gomoku.players.human import HumanPlayer
from gomoku.players.minimax import MinimaxPlayer


def _board_from(moves):
    b = Board()
    for r, c in moves:
        b.play(r, c)
    return b


def test_base_class_contract():
    assert isinstance(MinimaxPlayer("easy"), Player)
    assert MinimaxPlayer("hard").name == "minimax-hard"
    with pytest.raises(NotImplementedError):
        HumanPlayer().select_move(Board())


def test_empty_board_plays_center():
    for level in ("easy", "medium", "hard"):
        assert MinimaxPlayer(level).select_move(Board()) == 7 * 15 + 7


def test_extend_live_three_all_levels():
    # 黑活三 (7,6)(7,7)(7,8)，轮黑：正确应手是成活四 (7,5) 或 (7,9)
    b = _board_from([(7, 6), (1, 1), (7, 7), (3, 3), (7, 8), (5, 5)])
    for level in ("easy", "medium", "hard"):
        action = MinimaxPlayer(level).select_move(b)
        assert action in {7 * 15 + 5, 7 * 15 + 9}, f"{level} chose {action}"


def test_block_rush_four_all_levels():
    # 白有 (7,5)(7,6)(7,7) 连三 + (7,9)，轮黑：不堵 (7,8) 白下一手成五
    b = _board_from([(1, 1), (7, 5), (3, 3), (7, 6), (5, 5), (7, 7),
                     (9, 9), (7, 9)])
    for level in ("easy", "medium", "hard"):
        action = MinimaxPlayer(level).select_move(b)
        assert action == 7 * 15 + 8, f"{level} chose {action}"


def test_takes_immediate_win():
    # 黑 (7,5)(7,6)(7,7)(7,8) 活四待成，轮黑：直接连五
    b = _board_from([(7, 5), (1, 1), (7, 6), (3, 3), (7, 7), (5, 5), (7, 8), (9, 9)])
    action = MinimaxPlayer("hard").select_move(b)
    assert action in {7 * 15 + 4, 7 * 15 + 9}


def test_select_move_does_not_mutate_board():
    b = _board_from([(7, 7), (8, 8)])
    snapshot = b.grid.copy()
    MinimaxPlayer("easy").select_move(b)
    assert (b.grid == snapshot).all() and len(b.history) == 2


def test_rejects_finished_game():
    b = _board_from([(7, 4), (8, 0), (7, 5), (8, 1), (7, 6), (8, 2), (7, 7), (8, 3), (7, 8)])
    with pytest.raises(ValueError):
        MinimaxPlayer("easy").select_move(b)


def test_unknown_level():
    with pytest.raises(ValueError):
        MinimaxPlayer("lunatic")
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_minimax.py -q`
Expected: FAIL（`No module named 'gomoku.players.base'`）

- [ ] **Step 3: 写 base.py 与 human.py**

`src/gomoku/players/base.py`：

```python
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
```

`src/gomoku/players/human.py`：

```python
from gomoku.core import Board
from gomoku.players.base import Player


class HumanPlayer(Player):
    """真人。M0 中不自主思考——落子来自 Web UI 的请求。"""

    def __init__(self, name: str = "human"):
        self.name = name

    def select_move(self, board: Board) -> int:
        raise NotImplementedError("human moves come from the UI")
```

- [ ] **Step 4: 在 minimax.py 末尾追加搜索部分**

```python
from gomoku.core import Board  # 已有；确保也导入 BLACK（如需）
from gomoku.players.base import Player

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
        return best_action[0] * board.size + best_action[1]
```

注意：`from gomoku.players.base import Player` 放在文件顶部 import 区（文件上部为评估函数，无循环依赖：base.py 只依赖 core）。

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_minimax.py tests/test_evaluation.py -q`
Expected: `13 passed`。若 `test_block_rush_four_all_levels` 的 easy 档失败：easy 只看 1 层，靠候选排序的防守分堵四——若仍选错，把 `_window_score` 中对方视角分的权重并入 `point_score`（已并入）后复查模式表。

- [ ] **Step 6: Commit**

```bash
git add src/gomoku/players/base.py src/gomoku/players/human.py src/gomoku/players/minimax.py tests/test_minimax.py
git commit -m "feat: Player 抽象 + α-β 搜索三档 Minimax 专家系统"
```

---

### Task 6: Arena 裁判 + env 骨架

**Files:**
- Create: `src/gomoku/arena.py`
- Create: `src/gomoku/env.py`
- Test: `tests/test_arena.py`

**Interfaces:**
- Consumes: Task 2 `Board/BLACK/WHITE/DRAW`；Task 3 db 函数；Task 5 `Player/MinimaxPlayer`
- Produces:
  - `Arena(conn).play(black: Player, white: Player) -> dict`，返回 `{"game_id": int, "result": "black_win"|"white_win"|"draw", "moves": list}`；落子逐手入库，终局更新 games
  - `GomokuEnv`：接口骨架，方法抛 `NotImplementedError`（M2 实现）

- [ ] **Step 1: 写失败测试 tests/test_arena.py**

```python
from gomoku.arena import Arena
from gomoku.db import connect
from gomoku.players.minimax import MinimaxPlayer


def test_easy_vs_easy_completes_and_records(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    result = Arena(conn).play(MinimaxPlayer("easy"), MinimaxPlayer("easy"))
    assert result["result"] in {"black_win", "white_win", "draw"}
    assert len(result["moves"]) > 0
    (n,) = conn.execute(
        "SELECT COUNT(*) FROM moves WHERE game_id=?", (result["game_id"],)
    ).fetchone()
    row = conn.execute(
        "SELECT total_moves, result FROM games WHERE id=?", (result["game_id"],)
    ).fetchone()
    assert n == row["total_moves"] == len(result["moves"])
    assert row["result"] == result["result"]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_arena.py -q`
Expected: FAIL（`No module named 'gomoku.arena'`）

- [ ] **Step 3: 写 arena.py**

```python
from gomoku.core import Board, BLACK, WHITE, DRAW
from gomoku.db import create_game, record_move, finish_game
from gomoku.players.base import Player

_RESULT = {BLACK: "black_win", WHITE: "white_win", DRAW: "draw"}


class Arena:
    """任意两个 AI Player 的裁判：跑整局、逐手记录棋谱。

    真人不在 Arena 里（真人的落子来自 UI）；将来的基准赛
    （如 100 局交替先手对专家系统）直接复用本类。
    """

    def __init__(self, conn):
        self.conn = conn

    def play(self, black: Player, white: Player) -> dict:
        board = Board()
        game_id = create_game(self.conn, black.name, white.name)
        players = {BLACK: black, WHITE: white}
        while not board.game_over:
            player = players[board.current_player]
            action = player.select_move(board)
            row, col = divmod(action, board.size)
            board.play(row, col)
            record_move(self.conn, game_id, board.history[-1])
        finish_game(self.conn, game_id, _RESULT[board.winner], len(board.history))
        return {"game_id": game_id, "result": _RESULT[board.winner],
                "moves": list(board.history)}
```

- [ ] **Step 4: 写 env.py 骨架**

```python
"""Gym 风格训练环境骨架。M0 只冻结接口形状，M2（DQN 课）实现。

设计约定（见 docs/superpowers/specs/2026-09-05-m0-env-design.md）：
- 与对战平台共用 core.Board 规则内核，训练侧只是薄包装
- observation: np.float32 (2,15,15)，己方/对方平面（Board.observation）
- action: int 0..224（row*15+col）
- reward: 稀疏——胜 +1 / 负 -1 / 平 0 / 其余 0
- 对手机制（对手池/自对弈）M2 设计时定，不在此预埋
"""


class GomokuEnv:
    def __init__(self, opponent=None):
        raise NotImplementedError("M2 实现")

    def reset(self):
        """开新局，返回 observation。M2 实现。"""
        raise NotImplementedError("M2 实现")

    def step(self, action: int):
        """落子一手，返回 (observation, reward, terminated, truncated, info)。M2 实现。"""
        raise NotImplementedError("M2 实现")
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_arena.py -q`
Expected: `1 passed`（easy vs easy 全局，约几秒）

- [ ] **Step 6: Commit**

```bash
git add src/gomoku/arena.py src/gomoku/env.py tests/test_arena.py
git commit -m "feat: Arena 裁判（AI 对战+棋谱入库）+ GomokuEnv 接口骨架"
```

---

### Task 7: FastAPI 后端

**Files:**
- Create: `src/gomoku/server/__init__.py`（空文件）
- Create: `src/gomoku/server/app.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: Task 2/3/5/6 全部；`HumanPlayer`
- Produces（HTTP API，前端依赖）:
  - `POST /api/new` body `{black, white}` → 状态 JSON
  - `POST /api/move` body `{game_id, row, col}` → 状态 JSON（人走 + AI 应手各一手）
  - `POST /api/ai-move` body `{game_id}` → 状态 JSON（AI 走一手；机 vs 机由前端循环调用）
  - `GET /api/games` → 历史对局列表
  - 状态 JSON：`{game_id, board(15×15 二维数组), current_player, last_move, winner, game_over, moves}`
  - 若 `frontend/dist` 存在则挂载为静态根
  - 模块级 `app = FastAPI(...)`，`sessions: dict[int, dict]`（内存会话）

- [ ] **Step 1: 写失败测试 tests/test_server.py**

```python
from fastapi.testclient import TestClient

from gomoku.server.app import app

client = TestClient(app)


def test_new_game_human_vs_ai():
    r = client.post("/api/new", json={"black": "human", "white": "minimax-easy"})
    assert r.status_code == 200
    s = r.json()
    assert s["current_player"] == 1 and not s["game_over"]
    assert len(s["board"]) == 15 and len(s["board"][0]) == 15


def test_human_move_then_ai_reply():
    r = client.post("/api/new", json={"black": "human", "white": "minimax-easy"})
    gid = r.json()["game_id"]
    r = client.post("/api/move", json={"game_id": gid, "row": 7, "col": 7})
    assert r.status_code == 200
    s = r.json()
    assert s["board"][7][7] == 1
    assert len(s["moves"]) >= 2          # 人 + AI 应手
    assert s["board"][s["moves"][-1][1]][s["moves"][-1][2]] == 2


def test_invalid_move_400():
    r = client.post("/api/new", json={"black": "human", "white": "minimax-easy"})
    gid = r.json()["game_id"]
    client.post("/api/move", json={"game_id": gid, "row": 7, "col": 7})
    r = client.post("/api/move", json={"game_id": gid, "row": 7, "col": 7})
    assert r.status_code == 400          # 已占用
    r = client.post("/api/move", json={"game_id": 999999, "row": 7, "col": 7})
    assert r.status_code == 404          # 局不存在


def test_ai_vs_ai_via_ai_move_endpoint():
    r = client.post("/api/new", json={"black": "minimax-easy", "white": "minimax-easy"})
    s = r.json()
    gid, n = s["game_id"], 0
    while not s["game_over"] and n < 300:
        s = client.post("/api/ai-move", json={"game_id": gid}).json()
        n += 1
    assert s["game_over"]


def test_games_list():
    r = client.get("/api/games")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_server.py -q`
Expected: FAIL（`No module named 'gomoku.server.app'`）

- [ ] **Step 3: 写实现 src/gomoku/server/app.py**

```python
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gomoku.core import Board, BLACK, WHITE
from gomoku.db import connect, create_game, record_move, finish_game, list_games
from gomoku.players.base import Player
from gomoku.players.human import HumanPlayer
from gomoku.players.minimax import MinimaxPlayer

app = FastAPI(title="gomoku-rl")
conn = connect()
sessions: dict[int, dict] = {}   # game_id -> {"board": Board, "black": Player, "white": Player}

_RESULT = {BLACK: "black_win", WHITE: "white_win", 3: "draw"}


def _make_player(kind: str) -> Player:
    if kind == "human":
        return HumanPlayer()
    if kind.startswith("minimax-"):
        return MinimaxPlayer(level=kind.removeprefix("minimax-"))
    raise ValueError(f"unknown player kind: {kind}")


def _state(game_id: int) -> dict:
    s = sessions[game_id]
    b = s["board"]
    return {
        "game_id": game_id,
        "board": b.grid.tolist(),
        "current_player": b.current_player,
        "last_move": list(b.last_move) if b.last_move else None,
        "winner": b.winner,
        "game_over": b.game_over,
        "moves": [[p, r, c] for p, r, c in b.history],
    }


def _maybe_finish(game_id: int) -> None:
    board = sessions[game_id]["board"]
    if board.game_over:
        finish_game(conn, game_id, _RESULT[board.winner], len(board.history))


def _play_one_ai_move(game_id: int) -> bool:
    """当前行棋方若是 AI 则走一手并记录；返回是否走了。"""
    s = sessions[game_id]
    board = s["board"]
    if board.game_over:
        return False
    player = s["black"] if board.current_player == BLACK else s["white"]
    if isinstance(player, HumanPlayer):
        return False
    action = player.select_move(board)
    row, col = divmod(action, board.size)
    board.play(row, col)
    record_move(conn, game_id, board.history[-1])
    return True


class NewGameReq(BaseModel):
    black: str = "human"
    white: str = "minimax-hard"


class MoveReq(BaseModel):
    game_id: int
    row: int
    col: int


class GameIdReq(BaseModel):
    game_id: int


@app.post("/api/new")
def new_game(req: NewGameReq):
    board = Board()
    game_id = create_game(conn, req.black, req.white)
    try:
        sessions[game_id] = {
            "board": board,
            "black": _make_player(req.black),
            "white": _make_player(req.white),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _play_one_ai_move(game_id)   # 黑方是 AI 时先走一手
    return _state(game_id)


@app.post("/api/move")
def move(req: MoveReq):
    if req.game_id not in sessions:
        raise HTTPException(status_code=404, detail="game not found")
    board = sessions[req.game_id]["board"]
    if board.game_over:
        raise HTTPException(status_code=400, detail="game is over")
    player = (sessions[req.game_id]["black"] if board.current_player == BLACK
              else sessions[req.game_id]["white"])
    if not isinstance(player, HumanPlayer):
        raise HTTPException(status_code=400, detail="not human's turn")
    try:
        board.play(req.row, req.col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    record_move(conn, req.game_id, board.history[-1])
    _play_one_ai_move(req.game_id)   # AI 应手
    _maybe_finish(req.game_id)
    return _state(req.game_id)


@app.post("/api/ai-move")
def ai_move(req: GameIdReq):
    if req.game_id not in sessions:
        raise HTTPException(status_code=404, detail="game not found")
    _play_one_ai_move(req.game_id)
    _maybe_finish(req.game_id)
    return _state(req.game_id)


@app.get("/api/games")
def games():
    return list_games(conn)


# 生产模式：挂载前端构建产物（frontend/dist 存在时生效）
_DIST = pathlib.Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_server.py -q`
Expected: `5 passed`

- [ ] **Step 5: 手动起服冒烟**

Run: `uvicorn gomoku.server.app:app --port 8000`（后台），然后：

```bash
curl -s -X POST http://127.0.0.1:8000/api/new -H "Content-Type: application/json" -d '{"black": "human", "white": "minimax-easy"}'
```

Expected: 返回含 `game_id` 的状态 JSON。测完停掉进程。

- [ ] **Step 6: Commit**

```bash
git add src/gomoku/server/__init__.py src/gomoku/server/app.py tests/test_server.py
git commit -m "feat: FastAPI 后端（new/move/ai-move/games + 静态托管）"
```

---

### Task 8: React 前端脚手架 + 棋盘组件

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.js`, `frontend/index.html`
- Create: `frontend/src/main.jsx`, `frontend/src/api.js`, `frontend/src/Board.jsx`, `frontend/src/App.jsx`（本任务为最小可渲染版）, `frontend/src/index.css`

**Interfaces:**
- Consumes: Task 7 的 HTTP API（开发期经 Vite 代理）
- Produces: `npm run build` 产出 `frontend/dist`；`Board` 组件 props `{grid, lastMove, onPlay, locked}`；`api.newGame/move/aiMove/games`

- [ ] **Step 1: 写脚手架文件**

`frontend/package.json`：

```json
{
  "name": "gomoku-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.0"
  }
}
```

`frontend/vite.config.js`：

```js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
```

`frontend/index.html`：

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>gomoku-rl</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

`frontend/src/main.jsx`：

```jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
```

- [ ] **Step 2: 写 api.js**

`frontend/src/api.js`：

```js
async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  newGame: (black, white) => post("/api/new", { black, white }),
  move: (gameId, row, col) => post("/api/move", { game_id: gameId, row, col }),
  aiMove: (gameId) => post("/api/ai-move", { game_id: gameId }),
  games: () => fetch("/api/games").then((r) => r.json()),
};
```

- [ ] **Step 3: 写 Board.jsx（SVG 棋盘）**

`frontend/src/Board.jsx`：

```jsx
const N = 15, CELL = 36, PAD = 24, SIZE = PAD * 2 + CELL * (N - 1);

function xyToRC(x, y) {
  const c = Math.round((x - PAD) / CELL);
  const r = Math.round((y - PAD) / CELL);
  if (r < 0 || r >= N || c < 0 || c >= N) return null;
  const px = PAD + c * CELL, py = PAD + r * CELL;
  if (Math.hypot(x - px, y - py) > CELL * 0.45) return null;
  return [r, c];
}

export default function Board({ grid, lastMove, onPlay, locked }) {
  const handleClick = (e) => {
    if (locked) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const rc = xyToRC(e.clientX - rect.left, e.clientY - rect.top);
    if (rc && grid[rc[0]][rc[1]] === 0) onPlay(rc[0], rc[1]);
  };

  const lines = [];
  for (let i = 0; i < N; i++) {
    lines.push(
      <line key={"h" + i} x1={PAD} y1={PAD + i * CELL} x2={SIZE - PAD} y2={PAD + i * CELL}
            stroke="#8a6d3b" strokeWidth="1" />,
      <line key={"v" + i} x1={PAD + i * CELL} y1={PAD} x2={PAD + i * CELL} y2={SIZE - PAD}
            stroke="#8a6d3b" strokeWidth="1" />
    );
  }
  const stars = [[3, 3], [3, 11], [11, 3], [11, 11], [7, 7]];

  return (
    <svg data-testid="board" width={SIZE} height={SIZE} onClick={handleClick}
         style={{ background: "#dcb35c", cursor: locked ? "default" : "pointer",
                  borderRadius: 4 }}>
      {lines}
      {stars.map(([r, c]) => (
        <circle key={"s" + r + "-" + c} cx={PAD + c * CELL} cy={PAD + r * CELL}
                r="3" fill="#5a4632" />
      ))}
      {grid.map((row, r) =>
        row.map((v, c) =>
          v === 0 ? null : (
            <circle key={r + "-" + c} cx={PAD + c * CELL} cy={PAD + r * CELL}
                    r={CELL * 0.42} fill={v === 1 ? "#111" : "#fff"} stroke="#555" />
          )
        )
      )}
      {lastMove && grid[lastMove[0]][lastMove[1]] !== 0 && (
        <circle cx={PAD + lastMove[1] * CELL} cy={PAD + lastMove[0] * CELL}
                r="4" fill="#e33" />
      )}
    </svg>
  );
}
```

- [ ] **Step 4: 写最小 App.jsx 与样式（保证可构建）**

`frontend/src/App.jsx`（本任务最小版，Task 9 替换）：

```jsx
import Board from "./Board";

const EMPTY_GRID = Array.from({ length: 15 }, () => Array(15).fill(0));

export default function App() {
  return <Board grid={EMPTY_GRID} lastMove={null} onPlay={() => {}} locked />;
}
```

`frontend/src/index.css`：

```css
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
  background: #2b2b2b;
  color: #eee;
  display: flex;
  justify-content: center;
  min-height: 100vh;
}
.setup, .game { margin-top: 24px; text-align: center; }
.setup h1 { letter-spacing: 2px; }
.setup label { display: block; margin: 12px 0; font-size: 15px; }
button {
  padding: 8px 18px;
  border: 1px solid #666;
  border-radius: 6px;
  background: #444;
  color: #eee;
  cursor: pointer;
  font-size: 14px;
}
button:hover:not(:disabled) { background: #555; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
select {
  padding: 6px;
  border-radius: 6px;
  background: #444;
  color: #eee;
  border: 1px solid #666;
  margin-left: 8px;
}
.status {
  display: flex;
  align-items: center;
  gap: 16px;
  justify-content: center;
  margin-bottom: 10px;
  font-size: 16px;
}
.toolbar { margin-top: 14px; display: flex; gap: 10px; justify-content: center; }
.toolbar .pending { opacity: 0.7; font-size: 13px; }
.toast {
  position: fixed;
  bottom: 40px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.85);
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  pointer-events: none;
}
```

- [ ] **Step 5: 安装依赖并构建**

Run: `cd frontend && npm install && npm run build`
Expected: 构建成功，生成 `frontend/dist/`。`node_modules`/`dist` 已 gitignore。

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/vite.config.js frontend/index.html frontend/src/
git commit -m "feat: React 前端脚手架 + SVG 棋盘组件"
```

---

### Task 9: 对局流程与占位工具栏（App 完整版）

**Files:**
- Modify: `frontend/src/App.jsx`（整体替换）

**Interfaces:**
- Consumes: Task 8 `Board/api`；Task 7 API 契约
- Produces: 开局配置（黑白方任选 human/minimax-easy|medium|hard）→ 对局 → AI 自动应手（机 vs 机自动连走）→ 胜负提示；占位工具栏（悔棋/棋谱下载/历史对局/RL 对战/训练面板 → toast"后续开发"）

- [ ] **Step 1: 整体替换 frontend/src/App.jsx**

```jsx
import { useEffect, useRef, useState } from "react";
import Board from "./Board";
import { api } from "./api";

const OPTIONS = [
  { value: "human", label: "玩家" },
  { value: "minimax-easy", label: "Minimax·简单" },
  { value: "minimax-medium", label: "Minimax·中等" },
  { value: "minimax-hard", label: "Minimax·困难" },
];
const PENDING = [
  ["悔棋", "M1"], ["棋谱下载", "M1"], ["历史对局", "M1"],
  ["RL 对战", "M2+"], ["训练面板", "M4+"],
];

function labelOf(kind) {
  return OPTIONS.find((o) => o.value === kind)?.label ?? kind;
}

export default function App() {
  const [config, setConfig] = useState({ black: "human", white: "minimax-hard" });
  const [game, setGame] = useState(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");
  const toastTimer = useRef(null);

  const showToast = (msg) => {
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 2200);
  };

  const start = async () => {
    setBusy(true);
    try { setGame(await api.newGame(config.black, config.white)); }
    catch (e) { showToast(e.message); }
    finally { setBusy(false); }
  };

  const play = async (row, col) => {
    if (busy) return;
    setBusy(true);
    try { setGame(await api.move(game.game_id, row, col)); }
    catch (e) { showToast(e.message); }
    finally { setBusy(false); }
  };

  // AI 回合：延迟请求一手；机 vs 机时随状态更新自动连续走
  useEffect(() => {
    if (!game || game.game_over) return;
    const side = game.current_player === 1 ? config.black : config.white;
    if (side === "human") return;
    const t = setTimeout(async () => {
      setBusy(true);
      try { setGame(await api.aiMove(game.game_id)); }
      catch (e) { showToast(e.message); }
      finally { setBusy(false); }
    }, 400);
    return () => clearTimeout(t);
  }, [game]);   // eslint-disable-line react-hooks/exhaustive-deps

  if (!game) {
    return (
      <div className="setup">
        <h1>gomoku-rl</h1>
        <p>五子棋强化学习实战 · M0</p>
        {["black", "white"].map((side) => (
          <label key={side}>
            {side === "black" ? "黑方" : "白方"}：
            <select value={config[side]}
                    onChange={(e) => setConfig({ ...config, [side]: e.target.value })}>
              {OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
        ))}
        <div>
          <button onClick={start} disabled={busy}>开始对局</button>
        </div>
        {toast && <div className="toast">{toast}</div>}
      </div>
    );
  }

  const currentKind = game.current_player === 1 ? config.black : config.white;
  const humanTurn = currentKind === "human";

  return (
    <div className="game">
      <div className="status">
        {game.game_over
          ? (game.winner === 3
              ? "平局"
              : `胜者：${labelOf(game.winner === 1 ? config.black : config.white)}`)
          : `轮到：${labelOf(currentKind)}${busy && !humanTurn ? "（思考中…）" : ""}`}
        <span>第 {game.moves.length} 手</span>
        <button onClick={() => setGame(null)}>重新开局</button>
      </div>
      <Board grid={game.board} lastMove={game.last_move} onPlay={play}
             locked={busy || game.game_over || !humanTurn} />
      <div className="toolbar">
        {PENDING.map(([label, milestone]) => (
          <button key={label} className="pending"
                  onClick={() => showToast(`${label}：后续开发（${milestone}）`)}>
            {label}
          </button>
        ))}
      </div>
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
```

- [ ] **Step 2: 联调验证（双进程）**

终端 A：`uvicorn gomoku.server.app:app --port 8000`
终端 B：`cd frontend && npm run dev`

打开 `http://localhost:5173`，逐项检查：
1. 配置页可选黑白方，默认"玩家 vs Minimax·困难"
2. 人机局：点击落子 → AI 1~3 秒内应手 → 最后一手有红点标记
3. AI 思考期棋盘锁定；轮到玩家时才可点击
4. 机 vs 机局（黑方白方都选 Minimax）：自动连续走完全局并显示胜者
5. 任一档难度下完成一次五连 → 显示"胜者：xxx"
6. 点工具栏 5 个占位按钮 → toast 提示后续开发里程碑
7. "重新开局"回到配置页

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat: 对局流程（全模式选择/AI 自动应手/占位工具栏）"
```

---

### Task 10: 端到端验收 + 收尾

**Files:**
- Modify: `README.md`（勾选 M0）
- Test: 全量 pytest + 浏览器人工验收 + 查库

**Interfaces:**
- Consumes: 全部前序任务
- Produces: M0 完成的仓库状态

- [ ] **Step 1: 全量测试**

Run: `pytest -q`
Expected: 全部通过（test_smoke 1 + test_core 11 + test_db 2 + test_evaluation 5 + test_minimax 8 + test_arena 1 + test_server 5 = 33）

- [ ] **Step 2: 生产模式验收（构建产物直出）**

```bash
cd frontend && npm run build && cd ..
uvicorn gomoku.server.app:app --port 8000
```

浏览器打开 `http://127.0.0.1:8000`（FastAPI 直接托管 dist，不再需要 5173）：
- 完整下一局人 vs minimax-hard 并分出胜负
- 机 vs 机（medium vs medium）跑完一局

- [ ] **Step 3: 查库确认棋谱落库**

```bash
python -c "import sqlite3; c=sqlite3.connect('gomoku.db'); print(c.execute('SELECT id,black_name,white_name,result,total_moves FROM games ORDER BY id DESC LIMIT 5').fetchall()); print(c.execute('SELECT game_id,COUNT(*) FROM moves GROUP BY game_id ORDER BY game_id DESC LIMIT 5').fetchall())"
```

Expected: 最近 2 局 result 非 ongoing，moves 数与 total_moves 一致。

- [ ] **Step 4: 勾选 README 路线图 M0**

`README.md` 中 `- [ ] **M0** 五子棋环境 + Minimax 专家系统 + Web UI（人 vs 专家）` 改为 `- [x]`。

- [ ] **Step 5: Commit + push**

```bash
git add README.md
git commit -m "chore: M0 验收通过，路线图勾选 M0"
git push
```

---

## Self-Review 记录

1. **Spec 覆盖**：Board API/胜负/undo/observation/action 编码 → Task 2；SQLite 两表 + 扩展策略 → Task 3；评分表数值与模式 → Task 4；候选剪枝/α-β/三档/快检/天元 → Task 5；Player 抽象/HumanPlayer/Arena → Task 5/6；env 骨架 → Task 6；API 三端点 + games + 静态托管 → Task 7；React 全模式 + 占位工具栏 + 思考锁盘 → Task 8/9；验收三查（pytest/浏览器/查库）→ Task 10。spec"明确不做"清单均未实现。✓
2. **占位扫描**：无 TBD/TODO；所有代码步骤含完整代码。✓
3. **类型一致性**：`select_move(board) -> int`、`record_move(conn, game_id, (player,row,col))`、`finish_game(conn, game_id, result, total_moves)`、`candidates(board, limit=12) -> list[(r,c)]`、API 状态 JSON 字段在各任务间已比对一致；`Board.play` 终局时 `current_player` 停在最后一手方的约定在 `_search` 注释与 `_immediate` 中均有依据。✓
