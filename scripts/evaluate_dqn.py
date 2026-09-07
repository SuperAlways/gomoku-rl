"""课 01 · E1：基准赛——加载 checkpoint 打固定对手，衡量 RL 进步的固定标尺。

用法：
    python scripts/evaluate_dqn.py --checkpoint runs/e1/checkpoints/gen0059.pt \
        --opponent easy --n 100 --device auto
"""

import argparse

import torch

from gomoku.rl.dqn import evaluate, load_net, make_minimax_opponent, random_opponent


OPPONENTS = {
    "random": lambda: random_opponent,
    "easy": lambda: make_minimax_opponent("easy"),
    "medium": lambda: make_minimax_opponent("medium"),
    "hard": lambda: make_minimax_opponent("hard"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 基准赛")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", choices=list(OPPONENTS), default="easy")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_net(args.checkpoint, size=15, channels=64, device=device)
    opp = OPPONENTS[args.opponent]()
    rate = evaluate(net, opp, n=args.n, device=device)
    print(f"{args.opponent}: {rate:.2f} ({args.n} games)")


if __name__ == "__main__":
    main()
