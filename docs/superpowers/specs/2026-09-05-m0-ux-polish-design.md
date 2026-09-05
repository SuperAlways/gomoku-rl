# M0 UX 打磨设计：落子节奏 / 胜利特效 / 重开提示

日期：2026-09-05
状态：已与用户确认
范围：M0 收尾增量（feature/m0 分支，Task 10 端到端验收之前完成）

## 背景与动机

M0 人工验收反馈：基本可用，但三处体验需要打磨：
1. 人机局中玩家点击后"没反应"，片刻后玩家棋子与 AI 棋子一起出现——缺少"玩家先落、电脑后落"的节奏
2. 胜利无任何视觉强调
3. 终局后界面停滞，没有引导重开

## 1. 落子节奏：拆单手端点（已定方案）

后端 `/api/move` 改为只走人这一手，不再代 AI 应手；AI 落子统一走现有 `/api/ai-move`。人机局与机机局成为同一条单手循环：

```
点击 → POST /api/move（只走人这一手）→ 玩家棋子立即出现
     → 前端检测轮到 AI → POST /api/ai-move（400ms 延迟保留）→ AI 棋子出现（思考中指示）
```

- 后端：删除 move 端点中的 `_play_one_ai_move` 调用；校验（404 / 终局 400 / 非人回合 400 / 非法 400）、record_move、_maybe_finish 不变；`/api/new`、`/api/ai-move`、`/api/games` 不变
- 前端：App 的 `play()` 只发 `/api/move` 一手；AI 回合 useEffect（400ms 延迟 + aiMove）保持不变，人机局自动复用该路径。无乐观更新、无本地/服务器双状态
- 被否方案：前端乐观更新（后端不动但引入双状态与回滚复杂度）

## 2. 胜利五连字段：Board.winning_line()

- `core.Board` 新增 `winning_line() -> list[tuple[int, int]] | None`：以最后一手为中心，四方向找 ≥5 连的方向，返回包含最后一手的连续 5 格坐标（六连以上取含最后一手的一段 5 格）；未终局或平局返回 None
- `server/app.py` 的 `_state` 增加 `win_line` 字段：`list[list[int]] | None`（JSON 无元组）

## 3. 胜利特效：金色脉冲高亮（已定形态）

- Board 组件新增 prop `winLine`（`[[r,c],...] | null`）：命中的棋子加 `stone-win` class
- CSS 动画（纯 CSS，无新依赖）：
  - 金色光晕脉冲：`filter: drop-shadow` 呼吸
  - 轻微放大呼吸：`scale 1 → 1.15`（SVG circle 上用 `transform-box: fill-box; transform-origin: center`）
  - 依次点亮：`animation-delay = 序号 × 0.15s`
  - 其余棋子 opacity 降为 0.45，突出五连
- 被否形态：落子弹跳动画（复杂度高）、横幅+背景闪烁（效果粗糙）
- 平局无特效；特效与棋子渲染冲突时以特效优先

## 4. 终局 30 秒重开提示条（已定形态）

- 终局（含平局）后启动 30s 定时器；到时棋盘下方弹出提示条："再来一局？" + 两个按钮
  - 【重开一局】→ 回配置页（等同现有"重新开局"）
  - 【继续看】→ 关闭提示条，不再弹出；用户仍可随时手动"重新开局"
- 状态重置（重开/回配置页）时清理定时器；game_over 状态变化重新计时
- 被否形态：模态遮罩（太强制）、自动重开（用户可能还想看棋盘）

## 5. 测试与验收

- test_server.py：test_human_move_then_ai_reply 拆改——move 只走一手（moves 长度 1、current_player=2），AI 应手改由 /api/ai-move 驱动；新增 win_line 断言（人下到五连后 win_line 长度 5 且包含最后一手坐标）
- pytest 全量回归 + `npm run build`
- 浏览器人工验收：① 人机局落子节奏（玩家棋子即时出现，AI 棋子随后单独出现）② 金色脉冲特效与五连高亮 ③ 终局 30s 提示条两按钮行为

## 明确不做

- 乐观更新/双状态渲染
- 悔棋、棋谱下载等占位功能的实现（仍属 M1）
- 音效、粒子等更重的特效
