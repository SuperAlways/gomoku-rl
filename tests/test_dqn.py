import numpy as np
import pytest

torch = pytest.importorskip("torch")   # torch 未装则整模块跳过

from gomoku.rl.dqn import DQNNet, ReplayBuffer, SIZE  # noqa: E402


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


def test_replay_buffer_capacity_wraps():
    buf = ReplayBuffer(capacity=10)
    for i in range(25):
        buf.push(np.zeros((2, 15, 15), np.float32), i, 1.0,
                 np.ones((2, 15, 15), np.float32), False)
    assert len(buf) == 10


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    for i in range(8):
        buf.push(np.full((2, 15, 15), i, np.float32), i, 0.5,
                 np.full((2, 15, 15), i + 1, np.float32), False)
    states, actions, rewards, next_states, dones = buf.sample(8)
    assert states.shape == (8, 2, 15, 15)
    assert actions.shape == (8,) and actions.dtype == np.int64
    assert rewards.shape == (8,) and rewards.dtype == np.float32
    assert next_states.shape == (8, 2, 15, 15)
    assert dones.shape == (8,) and dones.dtype == bool