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