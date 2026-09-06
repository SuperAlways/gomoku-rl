import numpy as np
import pytest

torch = pytest.importorskip("torch")   # torch 未装则整模块跳过

from gomoku.rl.dqn import DQNNet, ReplayBuffer, SIZE, select_action  # noqa: E402
from gomoku.core import Board


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


def test_select_action_legal_and_masks_occupied():
    b = Board()
    b.play(7, 7)          # 黑
    b.play(7, 8)          # 白
    net = DQNNet(size=15, channels=8)
    # 把 (7,7) 对应动作的 Q 调成全网最高，但该位已占，必须被 mask 掉
    with torch.no_grad():
        net.head[-1].weight.zero_()
        net.head[-1].bias.zero_()
        net.head[-1].bias[7 * 15 + 7] = 10.0
        net.head[-1].bias[0] = 5.0            # 空位 (0,0) 次高
    a = select_action(net, b, eps=0.0)
    assert b.action_to_move(a) == (0, 0)      # (7,7) 已占，被迫选 (0,0)
    assert 0 <= a < 225


def test_select_action_eps_random_legal():
    b = Board()
    net = DQNNet(size=15, channels=8)
    for _ in range(30):
        a = select_action(net, b, eps=1.0)
        r, c = b.action_to_move(a)
        assert b.grid[r, c] == 0