import numpy as np
import pytest

torch = pytest.importorskip("torch")   # torch 未装则整模块跳过

from gomoku.rl.dqn import DQNNet, SIZE  # noqa: E402


def test_dqn_net_forward_shape():
    net = DQNNet(size=15, channels=8)
    obs = torch.zeros(4, 2, 15, 15)
    out = net(obs)
    assert out.shape == (4, 225)


def test_dqn_net_legal_positions_distinct():
    # 空盘上传一个两平面不一致的观测，输出对动作序号敏感（非全零）
    net = DQNNet(size=15, channels=8)
    obs = torch.zeros(1, 2, 15, 15)
    obs[0, 0, 7, 7] = 1.0                       # 自方一枚
    out = net(obs).squeeze(0)
    assert out.shape == (SIZE * SIZE,)
    assert (out != 0).any()