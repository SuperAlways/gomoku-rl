# M1 设计：对战平台定型——悔棋 / 历史回放 / SGF 下载 / Player 收尾

日期：2026-09-05
状态：已与用户逐节确认
范围：main 之后的 feature/m1 分支；README 路线图 M1 四项全做

## 背景与动机

M0 交付了环境 + Minimax 专家 + Web UI。M1 的目标是对战平台定型：把 M0 预留的占位按钮点亮，并为 M2 的 RL agent 铺平 Player 接口。核心原则延续 M0：**Board 是唯一状态权威**，前端（包括回放）只是视图。

已定决策（用户逐项确认）：
- 悔棋 = **悔一个回合**（人机局撤 AI 应手 + 自己那手；人人局撤一手）
- 历史回放 = **同页三视图**（配置 ↔ 对局 ↔ 历史/回放，不引入路由库）
- 回放执行层 = **前端本地重演**（一次性拉全谱，逐手本地渲染，零交互延迟）
- 棋谱下载 = **SGF** 单格式
- 权重加载预留 = **NNPlayer 骨架**（不做加载/推理实现）

## 1. 悔棋端点 `POST /api/undo`

请求 `{game_id}`，校验：404（无此局）/ 400（终局）/ 400（无可悔内容）。

回合语义按局型：
- 人机局：撤 2 手（AI 应手 + 人那手）；要求 moves ≥ 2 且当前轮到人（moves == 1 即黑 AI 先手，不可悔）
- 人人局：撤 1 手；要求 moves ≥ 1
- 机机局：400（无"你"可悔）

实现：`board.undo()` n 次 → `DELETE FROM moves WHERE game_id=? AND seq > len(history)` → commit → 返回 `_state`。`record_move` 以 `COUNT(*)` 推 seq，删行后自然衔接。终局不可悔，故 games 表 result/total_moves 无需回滚。

前端：对局页工具栏【悔棋】按钮，仅在满足上述回合条件时可用；点击后照常走现有单手循环（悔完轮到人，AI effect 不触发）。busy 期间锁。

## 2. 回放数据端点 `GET /api/games/{id}/record`

服务端从 moves 表重建一次 Board（顺带验证棋谱自洽），返回：

```json
{"game_id": 1, "black_name": "human", "white_name": "minimax-hard",
 "result": "black_win", "total_moves": 25,
 "moves": [[1,7,7],[2,7,8]], "win_line": [[7,7],[7,8]]}
```

`win_line` 为终局五连（复用 `Board.winning_line()`），前端回放到最后一手时显示金色特效——规则计算留在 core。查无此局 404。`GET /api/games` 列表端点已存在，直接复用。

## 3. SGF 导出

新模块 `src/gomoku/sgf.py`：`build_sgf(black_name, white_name, result, moves) -> str`。

- 坐标：`(row, col)` → SGF 两字母 a–o（(7,7)→"hh"，(0,0)→"aa"，(14,14)→"oo"）
- 头部：`SZ[15]`、`PB/PW` 双方名、`RE`（black_win→`B+`，white_win→`W+`，draw→`0`，ongoing→空）
- 着手序列：`;B[hh];W[ii]...` 按序输出

端点 `GET /api/games/{id}/sgf` → `text/x-sgf` 附件（`Content-Disposition: attachment; filename=gomoku-{id}.sgf`）。直接读库、不经 sessions——历史任意一局可下载。查无此局 404。

## 4. 前端：同页三视图

App 视图状态从"有无 game"扩展为三态（不引入路由库）：

- **配置页**：加【历史对局】按钮 → 历史视图
- **对局页工具栏**：悔棋 / 棋谱下载 / 历史对局三个按钮从占位变可用（RL 对战、训练面板仍 toast 占位）。棋谱下载 = 请求当前局 `/api/games/{id}/sgf` 触发浏览器下载；历史对局 = 切历史视图
- **历史视图**：对局列表（时间 / 黑方 / 白方 / 结果 / 手数），每行【回放】【SGF】，顶部【返回】回配置页
- **回放视图**：复用现有 `Board` 组件（只读、locked），本地状态 `ply`（0..total）；grid 由 `moves.slice(0, ply)` 本地重演；lastMove 高亮当前手；控制条【上一手】【下一手】【自动播放 500ms】【回到列表】；进度"第 x / 共 y 手"；播到最后一手用 record 的 `win_line` 显示金色特效

## 5. Player 接口收尾

- `src/gomoku/players/nn.py`：`NNPlayer(Player)` 骨架——`__init__(model_path: str | None = None)` 预留权重路径；`select_move` 抛 NotImplementedError；docstring 写明 M2 填什么（加载权重 → `observation()` → 前向 → mask 无效点 → 选 action）。风格与 `env.py` 骨架一致
- `players/__init__.py`：统一导出 `Player / HumanPlayer / MinimaxPlayer / NNPlayer`

## 6. 测试与验收

- **test_sgf.py**（新）：坐标映射三例（(7,7)→hh、(0,0)→aa、(14,14)→oo）、完整 SGF 含 SZ[15] 与全部着手、result 四种映射
- **test_server.py 追加**：
  - `test_undo_round`：人机局两轮后 undo → moves 回 2、棋盘对齐、库中该局行数同步
  - `test_undo_rejects`：终局 400 / 机机局 400 / hvA 仅 AI 先手 400
  - `test_record_endpoint`：404；正常局 moves 长度与 win_line 断言
  - `test_sgf_endpoint`：404；Content-Disposition 附件头；着手数一致
- **test_players.py**（新）：四类 Player 可从包根导入、NNPlayer 可带路径实例化
- **验收**：pytest 全量 + `npm run build` + 浏览器人工验收（悔棋回合效果 / 历史列表→回放→自动播放→特效 / SGF 下载文本可读）

## 明确不做（M1 范围外）

- 权重真实加载与前向推理（M2）
- 回放键盘快捷键、拖动进度条
- RL 对战 / 训练面板入口（仍 toast 占位，M2/M4）
