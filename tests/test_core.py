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
    # 配色 (r//2+c)%2：行内相邻异色、列内同色最多连 2、两斜同色最多连 2 ——
    # 全盘无任何方向五连。黑格恰 113 个 = 先手手数。
    # 落子顺序按"黑白格逐对交替"构造，使 play() 的轮流机制恰好复现该配色。
    black_cells = [(r, c) for r in range(15) for c in range(15) if (r // 2 + c) % 2 == 0]
    white_cells = [(r, c) for r in range(15) for c in range(15) if (r // 2 + c) % 2 == 1]
    assert len(black_cells) == 113 and len(white_cells) == 112
    b = Board()
    order = [cell for pair in zip(black_cells, white_cells) for cell in pair]
    order.append(black_cells[-1])
    for r, c in order:
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
