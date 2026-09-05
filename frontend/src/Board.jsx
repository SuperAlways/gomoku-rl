const N = 15, CELL = 36, PAD = 24, SIZE = PAD * 2 + CELL * (N - 1);

function xyToRC(x, y) {
  const c = Math.round((x - PAD) / CELL);
  const r = Math.round((y - PAD) / CELL);
  if (r < 0 || r >= N || c < 0 || c >= N) return null;
  const px = PAD + c * CELL, py = PAD + r * CELL;
  if (Math.hypot(x - px, y - py) > CELL * 0.45) return null;
  return [r, c];
}

export default function Board({ grid, lastMove, onPlay, locked }) {
  const handleClick = (e) => {
    if (locked) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const rc = xyToRC(e.clientX - rect.left, e.clientY - rect.top);
    if (rc && grid[rc[0]][rc[1]] === 0) onPlay(rc[0], rc[1]);
  };

  const lines = [];
  for (let i = 0; i < N; i++) {
    lines.push(
      <line key={"h" + i} x1={PAD} y1={PAD + i * CELL} x2={SIZE - PAD} y2={PAD + i * CELL}
            stroke="#8a6d3b" strokeWidth="1" />,
      <line key={"v" + i} x1={PAD + i * CELL} y1={PAD} x2={PAD + i * CELL} y2={SIZE - PAD}
            stroke="#8a6d3b" strokeWidth="1" />
    );
  }
  const stars = [[3, 3], [3, 11], [11, 3], [11, 11], [7, 7]];

  return (
    <svg data-testid="board" width={SIZE} height={SIZE} onClick={handleClick}
         style={{ background: "#dcb35c", cursor: locked ? "default" : "pointer",
                  borderRadius: 4 }}>
      {lines}
      {stars.map(([r, c]) => (
        <circle key={"s" + r + "-" + c} cx={PAD + c * CELL} cy={PAD + r * CELL}
                r="3" fill="#5a4632" />
      ))}
      {grid.map((row, r) =>
        row.map((v, c) =>
          v === 0 ? null : (
            <circle key={r + "-" + c} cx={PAD + c * CELL} cy={PAD + r * CELL}
                    r={CELL * 0.42} fill={v === 1 ? "#111" : "#fff"} stroke="#555" />
          )
        )
      )}
      {lastMove && grid[lastMove[0]][lastMove[1]] !== 0 && (
        <circle cx={PAD + lastMove[1] * CELL} cy={PAD + lastMove[0] * CELL}
                r="4" fill="#e33" />
      )}
    </svg>
  );
}
