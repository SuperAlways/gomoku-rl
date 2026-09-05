# gomoku-rl

> 一个以五子棋为载体的强化学习实战项目：从零搭建对战环境，亲手实现 DQN / REINFORCE / PPO / AlphaZero，让不同算法训练出的 AI 在同一个竞技场里对弈——同时以教程形式记录学习路上的每一个问题与坑。

A hands-on RL project built around Gomoku (Five-in-a-Row): build the environment from scratch, implement classic and modern RL algorithms yourself, and watch them battle in one arena. Written as a tutorial series so future learners can follow along.

---

## 为什么是五子棋？

学 RL，"看懂"和"动手"之间隔着一道鸿沟。CartPole 太简单感受不到真实问题，围棋/星际太大跑不动。五子棋恰好卡在中间：

- **规则人人秒懂**——不需要花时间理解环境本身，注意力全部留给 RL
- **规模适中**——15×15 棋盘，状态空间大到表格法失效（必须上神经网络），又小到一张消费级显卡能训出像样的 AI
- **自对弈可行**——环境就是对手自己，数据无限生成，这正是 AlphaGo 系列的立身之本
- **问题密度高**——稀疏奖励、非平稳对手、自对弈死循环、探索不足、评估失真……RL 的经典难题在这里全部会出现，而且**每一项都能在对弈中亲眼看到**

## 项目愿景

1. **一个可玩的对战平台**：人 vs 专家系统、人 vs 任意一代 RL 权重、RL vs RL 跨代对战、专家系统 vs AlphaZero——任意组合，浏览器里直接下
2. **一条算法进化史**：同一张棋盘上，Minimax → DQN → REINFORCE → PPO → AlphaZero 依次登场。每个算法都有权重档案和对专家系统的基准赛成绩单，算法之间的差距变成肉眼可见的胜率数字
3. **一套亲手写的 RL 代码**：所有算法从零实现（PyTorch），与 [hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl) 课程的章节互相印证——课程负责"为什么"，本项目负责"落到棋盘上是什么样"
4. **一份教程**：每课回答一个真实的 RL 学习问题，记录作者真实踩过的坑。后来者 clone 仓库、checkout 到某一课的 tag，就能看到"学到这一步时代码长什么样"

## 总体架构

```
┌───────────────────────────────────────────────────┐
│  gomoku-rl                                        │
│                                                   │
│  [环境层]  Board / 规则 / 胜负判定（纯 Python，可单测）│
│      ↓ Gym 风格接口                                │
│  [智能体层]  统一 Player 接口                       │
│      ├─ HumanPlayer（来自 UI）                     │
│      ├─ MinimaxPlayer（α-β 剪枝 + 棋型评估，3 档难度）│
│      └─ NNPlayer（加载 .pt 权重，算法无关）          │
│  [平台层]                                         │
│      ├─ Arena：任意两个 Player 对战 + 棋谱记录       │
│      ├─ Trainer：各算法自对弈训练脚本                │
│      └─ Web UI：FastAPI + 浏览器棋盘                │
└───────────────────────────────────────────────────┘
```

核心设计：**NNPlayer 不关心权重由哪个算法训出**——只要能回答"给定棋盘，每个格子几分"，就能登上竞技场。于是 DQN、PPO、AlphaZero 的产物全部进入同一个对战池。

专家系统（Minimax）与 RL agent 是平级的 Player。它既是开局第一个对手，也是衡量所有 RL 进步的**固定标尺**——自对弈的对手一直在变，但专家系统不变，定期和它打基准赛才能发现"自对弈胜率涨了、对固定对手却退化"这类经典陷阱。

## 课程线（每课两个锚点 tag）

| 课 | 标题——每一课回答一个核心问题 | 对应课程章节 | 洁净空间 | 成品 |
|---|---|---|---|---|
| 01 | 环境与专家系统——怎样把五子棋形式化成 MDP？ | ch03 MDP | `course/01-start` | `course/01` |
| 02 | DQN——当对手是会进步的自己，Q-learning 还成立吗？ | ch04 DQN | — | — |
| 03 | REINFORCE → A2C——稀疏奖励下策略梯度的方差有多大？ | ch05/06 | — | — |
| 04 | PPO——自对弈训练怎样才能稳定？ | ch07 PPO | — | — |
| 05 | AlphaZero——搜索与学习如何合流？ | ch32 自博弈 | — | — |
| 06 | 评估——为什么训练曲线会骗人？ | appendix | — | — |
| 附 | 踩坑附录：非平稳性 / 自对弈循环 / reward shaping 副作用 | common pitfalls | — | — |

每课固定结构：问题引入 → 实现（关键代码讲解）→ 动手实验（附"你应该会看到"的预期现象）→ 踩坑记录 → 与原课程的概念映射 → 基准赛考试。

### 如何跟随一门课

每门课在 git 历史上有两个锚点 tag：

- **`course/NN-start`（洁净空间）**：该课动工前的代码。clone 后
  `git checkout course/01-start`，跟着教程逐实验亲手重建——推荐的学习方式；
- **`course/NN`（结课成品）**：该课全部实验完成并通过结课考试后的代码。

每课由若干实验组成，每个实验以一个 `--no-ff` merge commit 的形式叠加在 main
上——想看"某个实验刚完成时代码长什么样"，checkout 那个 merge commit 即可。
分支与 tag 的完整规则见 `docs/course/branching-sop.md`。

## 路线图

- [x] **M0** 五子棋环境 + Minimax 专家系统 + Web UI（人 vs 专家）
- [x] **M1** 对战平台定型：Player 接口 / Arena / 棋谱存档回放 / 权重加载
- [ ] **M2** DQN agent：replay buffer、target network、对手池
- [ ] **M3** REINFORCE → baseline → A2C：亲眼看方差爆炸与它的解药
- [ ] **M4** PPO：自对弈稳定化、reward shaping 对比实验
- [ ] **M5** AlphaZero：MCTS + 策略价值网络，三方循环赛收官
- [ ] **M6** 训练档案：按代归档权重、训练曲线、跨代对战

## 算力拓扑

开发与对弈在本地（CPU 推理足够），训练脚本 headless 设计，主要训练在远程 GPU（RTX 5060 Ti 16GB）执行，权重文件在两机之间流转。所有权重按"代"归档——`gen001` 到 `gen200` 的对战，就是 RL 进步本身的样子。

## 致谢与参考

- [hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl) —— 本项目的概念源头，教程形态的范本
- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) —— Game/Coach/NNet/Arena 四层抽象的参考
- [lihongxun945/gobang](https://github.com/lihongxun945/gobang) —— Minimax 专家系统与棋型评估的中文教程
- [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) —— 五子棋 AlphaZero 的先行者

## License

TBD
