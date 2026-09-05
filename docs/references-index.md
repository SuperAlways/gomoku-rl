# 参考项目索引

> 本地路径：`references/`（gitignore，不入库）。开发时按"用途"列查找对应仓库，再按"关键文件"直奔主题，不必通读。

## 速查表

| 仓库 | 星数 | 语言 | 一句话定位 | 主要用途（对应里程碑） |
|---|---|---|---|---|
| `alpha-zero-general` | 4.5k | PyTorch | 通用 AlphaZero 教学框架 | 架构参考（M1/M5） |
| `AlphaZero_Gomoku` | 3.6k | Python/Keras | 专为五子棋写的 AlphaZero | MCTS+PV 网络精读（M5） |
| `LightZero` | 1.6k | PyTorch | OpenDILab 统一 MCTS 基准（NeurIPS'23） | 工程规范参考（M5 后） |
| `gobang-js` | 1.8k | JS | 五子棋 AI + 中文棋型评估教程 | **专家系统首选教材（M0）** |
| `gobang_AI` | 791 | Python | 博弈树 α-β 剪枝五子棋 | 专家系统 Python 参考（M0） |
| `gobang-html5` | 213 | JS | HTML5 五子棋人机对战 | UI 交互参考（M0） |
| `rapfi` | 263 | C++ | 世界顶级五子棋/连珠引擎 | 专家系统天花板，暂只留眼 |
| `AlphaZero_Gomoku_MPI` | 221 | Python/C++ | 并行化 AlphaGo Zero 五子棋 | 训练加速参考（M4/M5） |
| `alpha_zero` | 195 | PyTorch | 现代 AlphaZero（Go/Gomoku） | 现代 PyTorch 实现参考（M5） |
| `Alpha-Gobang-Zero` | 168 | PyTorch/Qt | RL 五子棋 + Qt 界面 | GUI 参考（可选） |

## 分仓库要点

### alpha-zero-general — 四层抽象的范本
- `Game.py` — 游戏抽象基类：`getInitBoard / getCanonicalForm / getActionSize / getValidMoves / getGameEnded / getNextState`
- `Coach.py` — 自对弈训练循环（episode → 训练样例 → 网络更新 → pit 对战评估新旧网络）
- `Arena.py` — 两个 agent 打 N 局的裁判
- `MCTS.py` — 带 Dirichlet 噪声的 MCTS
- `gobang/` — 自带五子棋实现（`gobang/GobangGame.py`、`gobang/pytorch/NNet.py`）
- `pretrained_models/` — 作者练好的各游戏权重，可直接加载对照
- **我们的借鉴**：Game/Player/Arena 接口划分，Coach 的"训练-评估"节拍

### AlphaZero_Gomoku — 最小可读实现
- `game.py` — 五子棋 Board 类（纯 Python，几百行）
- `mcts_pure.py` — 纯 MCTS（无网络），**M0 阶段就能跑，可当专家系统候选**
- `mcts_alphaZero.py` — 带 PV 网络的 MCTS
- `policy_value_net_pytorch.py` — 策略价值网络 PyTorch 版
- `human_play.py` — 人机对战入口
- 根目录有练好的 6×6 和 8×8 权重（`.model`），15×15 需自己训
- **我们的借鉴**：想最快理解 AlphaZero 全流程，从这四个文件读起

### gobang-js — 棋型评估的中文教材
- `src/ai/` — 核心 AI：评估函数、启发搜索、α-β 剪枝
- 配套中文教程见仓库 README（讲活三/冲四/眠三/禁手评分体系，系列文章）
- `src/minmax.worker.js` — Web Worker 里跑搜索（UI 不卡的经验）
- **我们的借鉴**：M0 的 Minimax 评估函数直接照此打分体系翻译成 Python

### gobang_AI — Python 版极简专家系统
- `gobang_AI.py` + `graphics.py`，单文件 α-β 剪枝，几百行
- **我们的借鉴**：Python 端骨架参考，逻辑比 JS 版简略

### gobang-html5 — 纯前端交互参考
- `src/` + `dist/`，无构建依赖可直接打开 `dist` 玩
- **我们的借鉴**：落子高亮、胜负提示、悔棋等交互细节

### LightZero — 工程化基准
- `lzero/envs/` — 内置 gomoku 等环境的 EnvAdapter 写法
- `lzero/policy/` — AlphaZero / MuZero 等策略实现
- `lzero/mcts/` — C++/Python 混合 MCTS（`ptree` 目录）
- **我们的借鉴**：环境注册与配置管理模式；MuZero 思路留作延伸阅读

### AlphaZero_Gomoku_MPI — 并行训练
- `train_mpi.py` — MPI 并行自对弈；`mcts_pure.py`/`mcts_alphaZero.py` 为改良版
- `model_15_15_5/` — **含 15×15 棋盘的模型与经验**，与我们尺寸一致，值得细看
- **我们的借鉴**：等上 5060 Ti 后若嫌训练慢，参考它的并行结构

### alpha_zero（michaelnny）— 现代 PyTorch 实现
- `alpha_zero/core/` — MCTS 与训练核心（含多个 mcts 版本对比）
- `games/` — Go/Gomoku 环境抽象
- `eval_play/` — 评估对局脚本
- ⚠️ 本地副本来自 tarball，`games/pro_games/` 下十几个文件名带冒号的职业棋谱在 Windows 上无法解出（代码文件完整）
- **我们的借鉴**：`checkpoint` 管理与评估脚本组织

### Alpha-Gobang-Zero — Qt 桌面版
- `alphazero/` — 算法；`app/` — PyQt 界面；`game.py` — 棋盘逻辑
- **我们的借鉴**：只在我们走桌面 UI 时翻；Web UI 路线下优先级最低

### rapfi — 天花板
- C++ 引擎 + 自带神经网络训练器（`Trainer/`）
- **我们的借鉴**：暂无。等 AlphaZero 跑通后，回来研究"顶级引擎强在哪"

## 网络环境备忘

- 本机对 GitHub HTTPS 不稳（`git clone` 常见 connection reset），**SSH 通道可用**（已认证 SuperAlways）
- 克隆命令模板：`git clone --depth 1 git@github.com:<owner>/<repo>.git`
- 真正失败的仓库用 tarball 兜底：`curl -sL https://codeload.github.com/<owner>/<repo>/tar.gz/refs/heads/main`
