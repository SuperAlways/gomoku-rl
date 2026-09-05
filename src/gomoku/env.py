"""Gym 风格训练环境骨架。M0 只冻结接口形状，M2（DQN 课）实现。

设计约定（见 docs/superpowers/specs/2026-09-05-m0-env-design.md）：
- 与对战平台共用 core.Board 规则内核，训练侧只是薄包装
- observation: np.float32 (2,15,15)，己方/对方平面（Board.observation）
- action: int 0..224（row*15+col）
- reward: 稀疏——胜 +1 / 负 -1 / 平 0 / 其余 0
- 对手机制（对手池/自对弈）M2 设计时定，不在此预埋
"""


class GomokuEnv:
    def __init__(self, opponent=None):
        raise NotImplementedError("M2 实现")

    def reset(self):
        """开新局，返回 observation。M2 实现。"""
        raise NotImplementedError("M2 实现")

    def step(self, action: int):
        """落子一手，返回 (observation, reward, terminated, truncated, info)。M2 实现。"""
        raise NotImplementedError("M2 实现")