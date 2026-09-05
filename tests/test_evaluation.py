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
    b = _board_from([(7, 6), (1, 2), (7, 7), (3, 5), (7, 8), (5, 1)])
    score_black = evaluate(b, BLACK)
    assert score_black >= 10_000          # 至少一个活三
    assert score_black > evaluate(b, 2)   # 黑方视角优于白方


def test_single_stones_score_zero():
    # 全是孤立单子时不该有棋型分
    b = _board_from([(1, 1), (3, 3), (5, 5), (7, 7), (9, 9), (11, 11)])
    assert evaluate(b, 1) == 0 and evaluate(b, 2) == 0


def test_point_score_ranks_extension():
    b = _board_from([(7, 6), (1, 2), (7, 7), (3, 5), (7, 8), (5, 1)])
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