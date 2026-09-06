import numpy as np
import pytest

torch = pytest.importorskip("torch")   # torch 未装则整模块跳过

from gomoku.rl.dqn import DQNNet, ReplayBuffer, SIZE, WIN_LEN, select_action, DQNTrainer, random_opponent, make_minimax_opponent, policy_entropy_from_logits, play_episode  # noqa: E402
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


def test_sync_target_copies_weights():
    net = DQNNet(size=15, channels=8)
    trainer = DQNTrainer(net, batch_size=16, device="cpu")
    w = trainer.net.head[-1].weight.data.clone()
    with torch.no_grad():
        trainer.target_net.head[-1].weight.fill_(0.0)
    trainer.sync_target()
    assert torch.equal(trainer.target_net.head[-1].weight.data, w)


def test_train_step_returns_loss_and_increments():
    net = DQNNet(size=15, channels=8)
    trainer = DQNTrainer(net, batch_size=16, device="cpu")
    for _ in range(16):
        trainer.buffer.push(*([np.zeros((2, 15, 15), np.float32), 0, 0.0,
                             np.zeros((2, 15, 15), np.float32), False]))
    loss = trainer.train_step()
    assert loss is not None and loss == loss           # 有限且非 NaN
    assert trainer.steps_done == 1


def test_eps_linear_then_clamped():
    trainer = DQNTrainer(DQNNet(size=15, channels=8), batch_size=16)
    assert trainer.get_eps(0, 1.0, 0.05, 100) == pytest.approx(1.0)
    mid = trainer.get_eps(50, 1.0, 0.05, 100)
    assert mid < 1.0 and mid > 0.05
    assert trainer.get_eps(200, 1.0, 0.05, 100) == pytest.approx(0.05)


def test_train_step_none_below_batch():
    trainer = DQNTrainer(DQNNet(size=15, channels=8), batch_size=16)
    assert trainer.train_step() is None


def test_random_opponent_legal():
    b = Board()
    a = random_opponent(b, np.random.default_rng(0))
    assert b.grid[b.action_to_move(a)] == 0


def test_minimax_opponent_returns_legal():
    opp = make_minimax_opponent("easy")
    b = Board(); b.play(7, 7)
    a = opp(b, np.random.default_rng(0))
    assert b.grid[b.action_to_move(a)] == 0


def test_policy_entropy_uniform_and_peaked():
    q = np.zeros(SIZE * SIZE)
    legal = np.arange(SIZE * SIZE)
    assert policy_entropy_from_logits(q, legal) == pytest.approx(np.log(SIZE * SIZE), rel=1e-3)
    q2 = np.zeros(SIZE * SIZE); q2[112] = 10.0
    assert policy_entropy_from_logits(q2, legal) < 0.1


def test_play_episode_sparse_reward_terminal():
    b = Board(size=SIZE, win_len=WIN_LEN)
    def fake_black(board):
        return int(np.random.choice(np.flatnonzero(board.valid_moves())))
    def fake_opponent(board):
        return int(np.random.choice(np.flatnonzero(board.valid_moves())))
    # 假 learner：持有 buffer 与 train_step
    class Fake:
        buf = None
        def __init__(self):
            self.buf = type("B", (), {"push": lambda *a, **k: None})()
            self.buffer = self.buf
        def train_step(self):
            return None
    info = play_episode(b, fake_black, fake_opponent, Fake())
    assert info["steps"] == len(b.history) > 0
    assert info["winner"] in {1, 2, 3}