from gomoku.core import Board, BLACK, WHITE, DRAW
from gomoku.db import create_game, record_move, finish_game
from gomoku.players.base import Player

_RESULT = {BLACK: "black_win", WHITE: "white_win", DRAW: "draw"}


class Arena:
    """任意两个 AI Player 的裁判：跑整局、逐手记录棋谱。

    真人不在 Arena 里（真人的落子来自 UI）；将来的基准赛
    （如 100 局交替先手对专家系统）直接复用本类。
    """

    def __init__(self, conn):
        self.conn = conn

    def play(self, black: Player, white: Player) -> dict:
        board = Board()
        game_id = create_game(self.conn, black.name, white.name)
        players = {BLACK: black, WHITE: white}
        while not board.game_over:
            player = players[board.current_player]
            action = player.select_move(board)
            row, col = divmod(action, board.size)
            board.play(row, col)
            record_move(self.conn, game_id, board.history[-1])
        finish_game(self.conn, game_id, _RESULT[board.winner], len(board.history))
        return {"game_id": game_id, "result": _RESULT[board.winner],
                "moves": list(board.history)}