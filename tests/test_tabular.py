import numpy as np
import pytest

from gomoku.core import Board, BLACK
from gomoku.rl.tabular import (
    blocker_opponent,
    load_qtable,
    policy_entropy,
    save_qtable,
    select_action,
    sliding_win_rate,
    state_key,
    td_backfill,
    train,
)


def test_state_key_distinguishes_player():
    grid = np.zeros((3, 3), np.int8)
    grid[0, 0] = BLACK
    grid[1, 1] = 2
    key_black = grid.tobytes() + bytes([BLACK])
    key_white = grid.tobytes() + bytes([2])
    assert key_black != key_white
    b = Board(size=3, win_len=3)
    b.play(0, 0)
    b.play(1, 1)          # 两手后轮回黑方，grid 与 key_black 一致
    assert state_key(b) == key_black


def test_blocker_blocks_threat():
    b = Board(size=3, win_len=3)
    b.play(0, 0)          # 黑
    b.play(2, 2)          # 白（blocker 方）
    b.play(0, 1)          # 黑成两连
    assert b.current_player == 2
    a = blocker_opponent(b, np.random.default_rng(0))
    assert b.action_to_move(a) == (0, 2)


def test_blocker_random_fallback():
    b = Board(size=3, win_len=3)
    a = blocker_opponent(b, np.random.default_rng(0))
    r, c = b.action_to_move(a)
    assert b.grid[r, c] == 0


def test_select_action_argmax_masks_occupied():
    b = Board(size=3, win_len=3)
    b.play(1, 1)          # 黑
    b.play(0, 0)          # 白
    key = state_key(b)
    q = {key: np.arange(9, dtype=float)}
    a = select_action(q, b, eps=0.0, rng=np.random.default_rng(0))
    assert b.action_to_move(a) == (2, 2)      # (0,0) 已占被 mask，argmax=8


def test_select_action_unknown_state_random_legal():
    b = Board(size=3, win_len=3)
    a = select_action({}, b, eps=0.0, rng=np.random.default_rng(0))
    assert 0 <= a < 9


def test_policy_entropy_uniform_and_peaked():
    legal = np.arange(9)
    assert policy_entropy(np.zeros(9), legal) == pytest.approx(np.log(9), rel=1e-3)
    peaked = np.zeros(9)
    peaked[4] = 10.0
    assert policy_entropy(peaked, legal) < 0.1


def test_td_backfill_values():
    s = np.zeros((3, 3), np.int8).tobytes() + bytes([BLACK])
    q = {s: np.zeros(9)}
    errs = td_backfill(q, [(s, 0), (s, 8)], reward=1.0, gamma=0.99, lr=0.5)
    assert q[s][8] == pytest.approx(0.5)                  # 终局手：0 + 0.5*(1-0)
    assert q[s][0] == pytest.approx(0.5 * 0.99 * 0.5)     # γ·max(合法 Q[s']) = 0.495
    assert errs == [pytest.approx(1.0), pytest.approx(0.495)]


def test_train_smoke_beats_random():
    # seed=0 实测最后 100 局胜率 ≈ 0.75，阈值留足余量
    q, rows = train(opponent="random", episodes=500, eps_start=1.0,
                    eps_end=0.1, eps_decay_episodes=300, gamma=0.99,
                    lr=0.5, seed=0)
    assert len(rows) == 500 and len(q) > 0
    assert rows[0]["episode"] == 0
    assert sliding_win_rate(rows)[-1] > 0.6


def test_save_load_roundtrip(tmp_path):
    q = {b"\x00" * 9 + b"\x01": np.arange(9, dtype=float)}
    path = tmp_path / "qtable.json"
    save_qtable(q, str(path), meta={"size": 3})
    q2, meta = load_qtable(str(path))
    assert meta["size"] == 3
    key = b"\x00" * 9 + b"\x01"
    assert np.allclose(q2[key], np.arange(9))


def test_sliding_win_rate():
    rows = [{"winner": 1}] * 150
    rates = sliding_win_rate(rows, window=100)
    assert len(rates) == 150 and rates[0] == 1.0 and rates[99] == 1.0
