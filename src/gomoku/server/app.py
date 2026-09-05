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
conn = connect(check_same_thread=False)
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
    line = b.winning_line()
    return {
        "game_id": game_id,
        "board": b.grid.tolist(),
        "current_player": b.current_player,
        "last_move": list(b.last_move) if b.last_move else None,
        "winner": b.winner,
        "game_over": b.game_over,
        "win_line": [[r, c] for r, c in line] if line else None,
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
    try:
        black = _make_player(req.black)
        white = _make_player(req.white)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    board = Board()
    game_id = create_game(conn, req.black, req.white)
    sessions[game_id] = {"board": board, "black": black, "white": white}
    _play_one_ai_move(game_id)   # 黑方是 AI 时先走一手
    _maybe_finish(game_id)
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
