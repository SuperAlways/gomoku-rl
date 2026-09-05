"""课 01 · E0：表格 Q-learning 训练入口（headless）。

用法：
    python scripts/train_tabular.py --opponent random --episodes 5000 \
        --eps-start 1.0 --eps-end 0.05 --eps-decay-episodes 3000 \
        --gamma 0.99 --lr 0.5 --exp-id e0-stage1

输出 runs/<exp-id>/：config.json + metrics.jsonl + curve.png + qtable.json
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gomoku.rl.tabular import save_qtable, sliding_win_rate, train


def plot_curves(rows, out_path: str, window: int = 100) -> None:
    episodes = [r["episode"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(episodes, sliding_win_rate(rows, window))
    axes[0, 0].set_title(f"win rate (sliding {window})")
    axes[0, 1].plot(episodes, [r["steps"] for r in rows])
    axes[0, 1].set_title("episode length")
    axes[1, 0].plot(episodes, [r["td_error"] for r in rows])
    axes[1, 0].set_title("TD error (mean |delta|)")
    axes[1, 1].plot(episodes, [r["entropy"] for r in rows])
    axes[1, 1].set_title("policy entropy")
    for ax in axes.flat:
        ax.set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="E0 表格 Q-learning 训练")
    parser.add_argument("--opponent", choices=["random", "blocker"],
                        default="random")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-episodes", type=int, default=3000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp-id", default="e0-stage1")
    args = parser.parse_args()

    out = pathlib.Path("runs") / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    q, rows = train(opponent=args.opponent, episodes=args.episodes,
                    eps_start=args.eps_start, eps_end=args.eps_end,
                    eps_decay_episodes=args.eps_decay_episodes,
                    gamma=args.gamma, lr=args.lr, seed=args.seed)

    with open(out / "metrics.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    plot_curves(rows, str(out / "curve.png"))
    save_qtable(q, str(out / "qtable.json"), meta={
        "size": 3, "win_len": 3, "opponent": args.opponent,
        "episodes": args.episodes, "gamma": args.gamma, "lr": args.lr,
    })
    print(f"done: {args.exp_id}  "
          f"final win rate (last 100) = {sliding_win_rate(rows)[-1]:.2f}")


if __name__ == "__main__":
    main()