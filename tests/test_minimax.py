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
    # 黑子特意选不共线布局——若黑方自己有一手冲四，depth-1 会正确地选择对攻而非堵四
    # 白有 (7,5)(7,6)(7,7) 连三 + (7,9)，轮黑：应堵 (7,8) 白下一手成五
    b = _board_from([(1, 1), (7, 5), (3, 4), (7, 6), (5, 2), (7, 7),
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
