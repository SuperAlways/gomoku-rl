import pytest

torch = pytest.importorskip("torch")

from gomoku.rl.dqn import DQNNet, evaluate, random_opponent


def test_evaluate_returns_winrate_bound():
    # 随机初始网络贪心对乱下，胜率应落在 [0,1]，且测试期间不训练网络
    net = DQNNet(size=15, channels=8)
    rate = evaluate(net, random_opponent, n=20, device="cpu")
    assert 0.0 <= rate <= 1.0
