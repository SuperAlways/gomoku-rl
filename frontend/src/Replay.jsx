import { useEffect, useRef, useState } from "react";
import Board from "./Board";

function buildGrid(moves, ply) {
  const grid = Array.from({ length: 15 }, () => Array(15).fill(0));
  for (let i = 0; i < ply; i++) {
    const [p, r, c] = moves[i];
    grid[r][c] = p;
  }
  return grid;
}

export default function Replay({ record, onBack }) {
  const { moves, win_line, black_name, white_name, result } = record;
  const [ply, setPly] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);

  // 自动播放：500ms 一手，播完自停
  useEffect(() => {
    if (!playing) return undefined;
    timer.current = setInterval(() => {
      setPly((p) => {
        if (p >= moves.length) {
          clearInterval(timer.current);
          setPlaying(false);
          return p;
        }
        return p + 1;
      });
    }, 500);
    return () => clearInterval(timer.current);
  }, [playing, moves.length]);

  const last = ply > 0 ? [moves[ply - 1][1], moves[ply - 1][2]] : null;
  const showWin = ply === moves.length && win_line ? win_line : null;

  return (
    <div className="game">
      <div className="status">
        {black_name} vs {white_name} · {result} · 第 {ply} / {moves.length} 手
      </div>
      <Board grid={buildGrid(moves, ply)} lastMove={last} winLine={showWin}
             onPlay={() => {}} locked />
      <div className="toolbar">
        <button onClick={() => setPly((p) => Math.max(0, p - 1))}
                disabled={playing || ply === 0}>上一手</button>
        <button onClick={() => setPly((p) => Math.min(moves.length, p + 1))}
                disabled={playing || ply === moves.length}>下一手</button>
        <button onClick={() => setPlaying((v) => !v)}
                disabled={moves.length === 0}>
          {playing ? "暂停" : "自动播放"}
        </button>
        <button onClick={onBack}>回到列表</button>
      </div>
    </div>
  );
}
