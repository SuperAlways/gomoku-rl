import base64
import json

from gomoku.core import Board
from gomoku.players.qtable import QTablePlayer


def _write_qtable(tmp_path, states) -> str:
    path = tmp_path / "qtable.json"
    path.write_text(json.dumps({"meta": {}, "states": states}),
                    encoding="utf-8")
    return str(path)


def test_select_move_argmax_masks_occupied(tmp_path):
    b = Board(size=3, win_len=3)
    b.play(1, 1)          # 黑
    b.play(0, 0)          # 白
    b.play(2, 2)          # 黑 → 轮白，查表的应是白方 Q
    key = b.grid.tobytes() + bytes([b.current_player])
    values = [0.0] * 9
    values[1] = 5.0       # (0,1) 最优；(0,0)(2,2) 已占会被 mask
    path = _write_qtable(tmp_path,
                         {base64.b64encode(key).decode(): values})
    p = QTablePlayer(path)
    assert b.action_to_move(p.select_move(b)) == (0, 1)


def test_select_move_unknown_state_random_legal(tmp_path):
    p = QTablePlayer(_write_qtable(tmp_path, {}))
    b = Board(size=3, win_len=3)
    r, c = b.action_to_move(p.select_move(b))
    assert b.grid[r, c] == 0