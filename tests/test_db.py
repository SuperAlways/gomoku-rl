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


def test_migrates_missing_size_column(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    raw = sqlite3.connect(path)
    raw.executescript(
        "CREATE TABLE games ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "started_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),"
        "black_name TEXT NOT NULL, white_name TEXT NOT NULL,"
        "result TEXT NOT NULL DEFAULT 'ongoing',"
        "total_moves INTEGER NOT NULL DEFAULT 0);"
    )
    raw.commit()
    raw.close()
    conn = connect(path)
    conn.execute("INSERT INTO games (black_name, white_name) VALUES ('a', 'b')")
    (size,) = conn.execute("SELECT size FROM games").fetchone()
    assert size == 15
    conn.close()


def test_create_game_with_size(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    gid = create_game(conn, "a", "b", size=3)
    (size,) = conn.execute("SELECT size FROM games WHERE id = ?", (gid,)).fetchone()
    assert size == 3
    conn.close()
