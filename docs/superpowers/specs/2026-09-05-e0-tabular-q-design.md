# E0 设计：表格 Q-learning（3×3 九宫格）——概念验收 + 网页对战

日期：2026-09-05
状态：已与用户逐节确认
范围：课 01 第一刀（feature/e0 分支）；实验清单见 docs/course/01-dqn-outline.md

## 背景与动机

课 01 的第一个实验 E0：在 3×3 连三制棋盘上跑**表格式 Q-learning**，把 ε-greedy、TD 更新、稀疏奖励回填整条环路在可亲眼看清的范围里跑通。训练成果（Q 表）通过 QTablePlayer 接入现有 Web UI，玩家可与它对战——Player 统一接入点的第一次兑现。

方法论遵循 docs/course/reference-homrl-methodology.md：观测层（指标四件套 + 健康/危险信号）、五段式实验笔记（问题→设置→你应该会看到→反例→结论）。

## 1. Board 泛化：`(size=15, win_len=5)`

- `core.Board` 构造签名改为 `Board(size: int = BOARD_SIZE, win_len: int = 5)`
- `_has_five` 更名 `_has_line(row, col, player)`，判定阈值用 `self.win_len`；`winning_line()` 中 `len(cells) >= 5` 改为 `>= self.win_len`、`cells[:5]` 改为 `cells[:self.win_len]`
- 默认参数下所有现有行为不变（全量测试守门）
- 不用 minimax 当 3×3 对手（其棋型打分为连五设计，3×3 下不可解释）

## 2. Q 表与训练脚本

- **状态键**：`board.grid.tobytes() + bytes([board.current_player])`——同一局面按行棋方区分
- **Q 表**：`dict[bytes, np.ndarray(size*size)]`，初始全 0；合法动作 mask（已占格不选）
- **训练循环**（`scripts/train_tabular.py`，headless CLI）：
  - `--opponent random|blocker --episodes N --eps-start --eps-end --eps-decay-episodes --gamma 0.99 --lr 0.5 --exp-id e0-stage1`
  - ε-greedy 线性退火；每局终局统一回填奖励（胜 +1 / 负 -1 / 平 0，只回填 Q 学员的着）
  - TD 更新：`Q[s][a] += lr * (r + γ * max_a' Q[s'][a'] * (1-done) - Q[s][a])`，episode 内逐手反向
- **对手**：
  - `random`：均匀随机合法位
  - `blocker`：贪心脚本——对方两连则必堵，否则随机（固定不可学）
  - 学员固定执黑先行（3×3 先手优势大，教学上先只学先手）
- **输出**（`runs/<exp-id>/`）：`config.json` + `metrics.jsonl`（逐局：episode/winner/steps/eps/td_error/entropy）+ `curve.png`（胜率滑动 100/平均局长/TD error/策略熵 四联图）+ `qtable.json`（最终 Q 表，网页对战加载用）

## 3. 观测层指标（健康/危险信号）

| 指标 | 健康 | 危险 |
|------|------|------|
| 胜率（滑动 100 局） | 稳步上升后平台 | 长期不动/回落 |
| 平均局长 | 先升后降 | 无限拉长 |
| TD error 均值 | 大→收敛变小 | 不收敛/回弹 |
| 策略熵（Q 表导出） | 缓慢下降 | 骤降 |

`scripts/plot_qtable.py <qtable.json>`：打印/绘制 3×3 每格 Q 值热力图（开局局面、双威胁局面两个快照）。

## 4. 网页对战接入

- **QTablePlayer**（`src/gomoku/players/qtable.py`）：实现 `Player`；`__init__(qtable_path: str, name="qtable-3x3")` 加载 JSON；`select_move`：查表 + mask 非法位 + argmax（查不到的局面→随机合法位）；`--human` 落子路径不涉及
- **server**：`/api/new` 请求增加可选 `size`（默认 15）；`_make_player` 加 `qtable` 分支（加载 `runs/<exp-id>/qtable.json` 路径约定）；棋盘 `Board(size=req.size, win_len=3 if req.size == 3 else 5)`
- **前端**：
  - `Board.jsx` 参数化：`N`/`CELL`/`PAD` 由 props `size` 推导（默认 15），SVG 尺寸与星位点（15 才画星位）随之自适应
  - 配置页 OPTIONS 增加「九宫格·Q表」单一选项：选中即 3×3 棋盘、黑方 qtable、白方 human（不另设独立棋盘尺寸选择器，避免两套入口）
  - `api.js`：newGame 透传 size
- 回放/SGF/悔棋对 3×3 局自动兼容（均按 grid/history 通用逻辑写）

## 5. 实验笔记与验收

- `docs/course/01-dqn/e0-tabular.md`：五段式笔记，两阶段各一组真实曲线 + Q 表热力图快照
- **验收标准**：
  1. 阶段 1（vs random）：胜率从 ~50%（先手随机对随机）升至 90%+
  2. 阶段 2（vs blocker）：学会双威胁（堵一漏一）取胜，胜率显著上升
  3. Q 表热力图可见天元/双威胁点价值结构
  4. 网页上选「九宫格·Q表」对战：AI 开局下天元、能赢会堵
  5. 全量 pytest 通过 + npm run build 通过

## 明确不做（E0 范围外）

- 15×15 DQN（第二刀单独 spec）
- Q 表压缩/函数近似（课 01 主体讲"表格的极限"时用 3×3 数据说明）
- 学员执白训练（先手/后手对称性留作讨论点）
- ELO 积分（对手只有一个，无相对强度可言）
