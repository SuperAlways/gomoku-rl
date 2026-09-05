"""课 01 · E0：Q 表热力图快照。

用法：python scripts/plot_qtable.py runs/e0-stage1/qtable.json

打印并绘制两个局面（空盘·黑先 / 双威胁局面·黑先）的 3×3 Q 值，
输出 PNG 与 qtable.json 同目录（qtable-snapshots.png）。
局面不在 Q 表中时打印警告并以全 0 兜底。
"""

import base64
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gomoku.core import BLACK


def _empty_key() -> bytes:
    return np.zeros((3, 3), dtype=np.int8).tobytes() + bytes([BLACK])


def _double_threat_key() -> bytes:
    """黑 (0,0)(2,0)(1,1)、白 (0,1)(1,0)、黑先：两个威胁点 (0,2) 与 (2,2)。"""
    grid = np.zeros((3, 3), dtype=np.int8)
    for r, c in [(0, 0), (2, 0), (1, 1)]:
        grid[r, c] = 1
    for r, c in [(0, 1), (1, 0)]:
        grid[r, c] = 2
    return grid.tobytes() + bytes([BLACK])


def _draw(ax, values, title: str) -> None:
    ax.imshow(values.reshape(3, 3), vmin=-1, vmax=1, cmap="RdYlGn")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{values[r * 3 + c]:.2f}",
                    ha="center", va="center", fontsize=12)
    ax.set_title(title)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    q = {base64.b64decode(k): np.asarray(v)
         for k, v in payload["states"].items()}
    snapshots = [
        ("开局：空盘·黑先", _empty_key()),
        ("双威胁局面·黑先", _double_threat_key()),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, (title, key) in zip(axes, snapshots):
        values = q.get(key)
        if values is None:
            print(f"[警告] Q 表中没有该局面：{title}")
            values = np.zeros(9)
        print(f"\n{title}\n{values.reshape(3, 3)}")
        _draw(ax, values, title)
    fig.tight_layout()
    out = path.parent / "qtable-snapshots.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()