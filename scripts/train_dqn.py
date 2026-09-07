"""课 01 · E1：完整配方 DQN 训练入口（headless，A2 快门派）。

用法：
    python scripts/train_dqn.py --epochs 60 --episodes-per-era 20000 \
        --batch-size 256 --device auto --exp-id e1

输出 runs/e1/：config.json + metrics.jsonl + curve.png + checkpoints/genNNNN.pt
"""

import argparse
import json
import pathlib

import torch

from gomoku.rl.dqn import DQNNet, DQNTrainer, train, plot_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 完整配方 DQN 训练")
    parser.add_argument("--epochs", type=int, default=60)          # 即 num_eras
    parser.add_argument("--episodes-per-era", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=720_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exp-id", default="e1")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path("runs") / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    net = DQNNet(size=15, channels=64)
    trainer = DQNTrainer(net, gamma=args.gamma, lr=args.lr,
                         batch_size=args.batch_size, target_sync_steps=10_000,
                         device=device)
    metrics = train(trainer, episodes_per_era=args.episodes_per_era,
                    num_eras=args.epochs, eps_start=args.eps_start,
                    eps_end=args.eps_end, eps_decay=args.eps_decay,
                    seed=args.seed, device=device, out_dir=str(out))
    plot_curves(metrics, str(out / "curve.png"))
    last = metrics[-1]
    print(f"done: {args.exp_id}  win_vs_easy={last['win_vs_easy']:.2f} "
          f"win_vs_random={last['win_vs_random']:.2f}")


if __name__ == "__main__":
    main()
