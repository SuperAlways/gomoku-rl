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


def connect(path: str = "gomoku.db", check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
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
