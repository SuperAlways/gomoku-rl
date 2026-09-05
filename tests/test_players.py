import pytest

from gomoku.core import Board
from gomoku.players import HumanPlayer, MinimaxPlayer, NNPlayer, Player


def test_player_exports():
    assert issubclass(HumanPlayer, Player)
    assert issubclass(MinimaxPlayer, Player)
    assert issubclass(NNPlayer, Player)


def test_nn_player_skeleton():
    p = NNPlayer(model_path="weights/gen001.pt")
    assert p.model_path == "weights/gen001.pt"
    assert p.name == "nn"
    with pytest.raises(NotImplementedError):
        p.select_move(Board())