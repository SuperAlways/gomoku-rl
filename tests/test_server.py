import pytest
from fastapi.testclient import TestClient

from gomoku.db import connect
from gomoku.server import app as app_module

client = TestClient(app_module.app)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """每个测试使用独立的临时数据库，并清空 sessions 字典。"""
    db_path = str(tmp_path / "test.db")
    new_conn = connect(db_path, check_same_thread=False)
    old_conn = app_module.conn
    old_sessions = app_module.sessions
    monkeypatch.setattr(app_module, "conn", new_conn)
    app_module.sessions = {}
    try:
        yield
    finally:
        app_module.conn = old_conn
        app_module.sessions = old_sessions
        new_conn.close()


def test_new_game_human_vs_ai():
    r = client.post("/api/new", json={"black": "human", "white": "minimax-easy"})
    assert r.status_code == 200
    s = r.json()
    assert s["current_player"] == 1 and not s["game_over"]
    assert len(s["board"]) == 15 and len(s["board"][0]) == 15


def test_human_move_single_ply():
    r = client.post("/api/new", json={"black": "human", "white": "minimax-easy"})
    gid = r.json()["game_id"]
    r = client.post("/api/move", json={"game_id": gid, "row": 7, "col": 7})
    assert r.status_code == 200
    s = r.json()
    assert s["board"][7][7] == 1
    assert len(s["moves"]) == 1            # 只走人这一手
    assert s["current_player"] == 2
    # AI 应手改由 /api/ai-move 驱动
    s = client.post("/api/ai-move", json={"game_id": gid}).json()
    assert len(s["moves"]) == 2 and s["current_player"] == 1


def test_win_line_field():
    r = client.post("/api/new", json={"black": "human", "white": "human"})
    gid = r.json()["game_id"]
    for row, col in [(7, 4), (8, 0), (7, 5), (8, 1), (7, 6), (8, 2), (7, 7), (8, 3), (7, 8)]:
        s = client.post("/api/move", json={"game_id": gid, "row": row, "col": col}).json()
    assert s["game_over"] and s["winner"] == 1
    assert len(s["win_line"]) == 5
    assert [7, 8] in s["win_line"] and [7, 4] in s["win_line"]


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
    r = client.post("/api/new", json={"black": "human", "white": "minimax-easy"})
    assert r.status_code == 200
    gid = r.json()["game_id"]
    r = client.get("/api/games")
    assert r.status_code == 200
    games = r.json()
    assert isinstance(games, list)
    match = [g for g in games if g["id"] == gid]
    assert len(match) == 1
    assert match[0]["black_name"] == "human"
    assert match[0]["white_name"] == "minimax-easy"


def test_new_game_rejects_unknown_kind():
    r = client.post("/api/new", json={"black": "gpt-5", "white": "human"})
    assert r.status_code == 400
