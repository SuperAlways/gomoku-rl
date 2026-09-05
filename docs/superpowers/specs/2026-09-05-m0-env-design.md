# M0 设计：五子棋环境 + Minimax 专家系统 + Web UI

日期：2026-09-05
状态：已与用户逐节确认

## 目标

M0 一次性交付三件套：规则内核、Minimax 专家系统（3 档难度）、全模式 Web UI。环境接口从第一天起同时服务"人对战平台"与后续 RL 训练（M2 起点亮）。最终形态是一个全量五子棋程序，未实现的功能在 UI 上占位并提示后续开发。

已定决策（承接 README 与本次讨论）：
- 15×15，无禁手自由规则
- src/ 布局；Board 核心采用 AlphaZero_Gomoku 风格 + numpy 表示
- Minimax 难度 = 搜索深度分档（easy/medium/hard = depth 1/2/4）
- 评分体系翻译自 lihongxun945/gobang（gobang-js）的 shape.js/eval.js
- 前端 Vite + React；后端 FastAPI；存储 SQLite
- Board 是唯一状态权威，前端只是视图

## 1. 总体架构与包结构

```
gomoku-rl/
├── src/gomoku/
│   ├── __init__.py
│   ├── core.py          # Board：规则内核（唯一的状态权威）
│   ├── db.py            # SQLite 存储（games / moves 表）
│   ├── env.py           # Gym 风格训练环境（M0 只留签名骨架，M2 点亮）
│   ├── arena.py         # Arena：跑整局 + 写棋谱
│   ├── players/
│   │   ├── base.py      # Player 抽象
│   │   ├── human.py     # 真人落子来自 UI 请求
│   │   └── minimax.py   # MinimaxPlayer，三档
│   └── server/
│       ├── app.py       # FastAPI
│       └── ...          # 挂载 frontend/dist
├── frontend/            # Vite + React
├── tests/
│   ├── test_core.py
│   ├── test_minimax.py
│   └── test_arena.py
└── docs/
```

数据流（人对战一局）：
浏览器点击 → `POST /api/move {game_id, row, col}` → FastAPI 校验 → `Board.play()` → 若未终局，请求 `MinimaxPlayer.select_move(board)` 得到应手并再次 `Board.play()` → 返回局面 JSON → React 渲染。一次往返完成人+AI 两手，前端无需轮询。

关键原则：Board 是唯一状态权威。后端每局持有 Board 实例 + session（内存，M0 不做持久会话）；前端不维护棋局逻辑，避免前后端两套规则实现。

## 2. Board 核心（core.py）

借鉴 `AlphaZero_Gomoku/game.py` 风格，改动点如下：

- 棋盘表示：`numpy int8 (15,15)`，0 空 / 1 黑 / 2 白。落子历史另存 `list[(player, row, col)]`。
  （不采用原仓库的 dict 表示：numpy 对 NN 输入与 Minimax 评估更友好。）
- 核心 API：
  - `play(row, col)` — 落子、切换 current_player、自动判定胜负
  - `valid_moves()` — 15×15 bool 合法位
  - `winner` — 0 进行中 / 1 黑 / 2 白 / 3 平局（满盘）
  - `undo()` — 弹出最后一手（UI 悔棋、自对弈回滚共用）
  - `observation()` — `(2,15,15)` 特征平面：己方/对方棋子；历史通道留 M2 扩展
  - `move_to_action(r,c)` / `action_to_move(a)` — action = r*15+c，RL 全程用整数 action
- 胜负判定：以最后一手落点为中心查 4 方向线性连五（不做全盘扫描）。无禁手。
- 依赖：仅 Python + numpy，逻辑全部可单测。

## 3. 存储（db.py）

- 标准库 `sqlite3`，单文件 `gomoku.db`（gitignore）。
- M0 两张表：
  - `games`: id, started_at, black_name, white_name, result(black_win/white_win/draw/ongoing), total_moves
  - `moves`: game_id, seq, player, row, col —— 对局过程中逐手写入
- Arena 负责写库；Web 后端与将来基准赛脚本走同一条路。
- 扩展策略：以后加新表（基准赛成绩、按代权重档案），不动 M0 的表。
- 边界：SQLite 只管棋谱与成绩。RL 训练样本（大规模 transition）M2 起走内存 replay buffer / 文件，不进库。

选择 SQLite 的理由：标准库零部署；"可查询"是 M6 训练档案刚需（如按代查对专家胜率）。前端"历史对局"功能即查这两张表。

## 4. Minimax 专家系统（players/minimax.py）

- 棋型识别与评分：以候选点为中心取 4 方向各 9 格窗口，按 gobang-js 打分体系评分：
  - 连五 10_000_000、活四 1_000_000、冲四 100_000、活三 10_000、眠三 1_000、活二 500、眠二 100
  - 双威胁叠加组合（如冲四+活三 ≈ 必胜型）——棋力的主要来源
- 候选剪枝：只考虑已有棋子 2 格切比雪夫邻域内的空点，按启发分排序取前 N（约 10~15 个）
- α-β 主循环：翻译 gobang-js `minmax.js`（130 行）骨架；叶子节点 = 我方总分 − 2×对方总分（对方分加权，防守优先）
- 难度分档：
  - easy = depth 1；medium = depth 2；hard = depth 4 + 顶层"立即取胜/必挡"快检
- 边角规则：空盘下天元。M0 不做强制思考限时——depth 4 纯 Python 最坏约 1~3 秒，可接受；实测过慢再做迭代加深
- 实例 `name` 自动带档位（`minimax-hard`），入库以此区分对手

## 5. Player / Arena

```python
# players/base.py
class Player(ABC):
    name: str
    @abstractmethod
    def select_move(self, board: Board) -> int: ...
```

- `HumanPlayer`：M0 中不自主思考，UI 落子请求即其 select_move
- `MinimaxPlayer(depth)`：如上
- `Arena.play(black, white, record=True)`：循环 select_move → board.play() → 每手写 moves 表 → 终局更新 games 表，返回棋谱。它是将来基准赛（100 局、交替先手）的唯一裁判
- `env.py`：M0 只写 `GomokuEnv` 骨架（reset/step/observation/reward 签名 + docstring 说明 M2 填什么），兑现分层承诺但不提前实现

## 6. Web UI

### 后端（server/app.py）

- `POST /api/new` — `{black, white, difficulty}` 建局，插入 games 行，返回 `{game_id, board, current_player}`
- `POST /api/move` — `{game_id, row, col}`：校验轮次与合法性 → 人落子 → AI 应手 → 返回 `{board, last_move, winner, game_over}`
- `GET /api/games` — 历史对局列表（接口先通，页面按钮置灰）
- FastAPI 挂 `frontend/dist`；开发时 Vite dev server 代理 `/api`

### 前端（frontend/，Vite + React）

- `Board` 组件：15×15 线格（Canvas 或 CSS Grid），点击发坐标；最后一手高亮；终局五连高亮
- 对局面板：双方名称/执子色、当前手、手数、重开一局
- 模式选择：开局前为黑白方各选 `human / minimax-easy / minimax-medium / minimax-hard` 任意组合 → 人vs人、人vs机、机vs机全模式成立（机vs机时前端循环请求即可）
- 愿景占位工具栏：悔棋、棋谱下载、历史对局/回放、RL 对战（选权重）、训练面板——未实现的点击 toast"后续开发（M1/M6/…）"；界面形态一步到位，功能逐课点亮
- AI 思考指示器；AI 思考期间锁盘

## 7. 测试与验收

- test_core.py：横/竖/两斜连五、满盘平局、undo 后状态一致、action↔move 往返、非法落子拒绝
- test_minimax.py：活三残局三档均出正确应手；hard 能堵对方冲四
- test_arena.py：minimax-easy vs minimax-easy 完整终局且库中 moves 完整
- 验收：浏览器人 vs minimax-hard 完整一局 + `sqlite3 gomoku.db` 查到该局棋谱

## 明确不做（M0 范围外）

- 悔棋 UI（Board.undo 已备，M1 接）
- 棋谱下载/回放页面（表结构已备，M1 接）
- RL 训练（M2+）、MCTS（M5）
- Zobrist 哈希缓存、并行搜索等性能优化
