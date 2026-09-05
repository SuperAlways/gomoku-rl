import { useEffect, useRef, useState } from "react";
import Board from "./Board";
import { api } from "./api";

const OPTIONS = [
  { value: "human", label: "玩家" },
  { value: "minimax-easy", label: "Minimax·简单" },
  { value: "minimax-medium", label: "Minimax·中等" },
  { value: "minimax-hard", label: "Minimax·困难" },
];
const PENDING = [
  ["悔棋", "M1"], ["棋谱下载", "M1"], ["历史对局", "M1"],
  ["RL 对战", "M2+"], ["训练面板", "M4+"],
];

function labelOf(kind) {
  return OPTIONS.find((o) => o.value === kind)?.label ?? kind;
}

export default function App() {
  const [config, setConfig] = useState({ black: "human", white: "minimax-hard" });
  const [game, setGame] = useState(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");
  const toastTimer = useRef(null);

  const showToast = (msg) => {
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 2200);
  };

  const start = async () => {
    setBusy(true);
    try { setGame(await api.newGame(config.black, config.white)); }
    catch (e) { showToast(e.message); }
    finally { setBusy(false); }
  };

  const play = async (row, col) => {
    if (busy) return;
    setBusy(true);
    try { setGame(await api.move(game.game_id, row, col)); }
    catch (e) { showToast(e.message); }
    finally { setBusy(false); }
  };

  // AI 回合：延迟请求一手；机 vs 机时随状态更新自动连续走
  useEffect(() => {
    if (!game || game.game_over) return;
    const side = game.current_player === 1 ? config.black : config.white;
    if (side === "human") return;
    const t = setTimeout(async () => {
      setBusy(true);
      try { setGame(await api.aiMove(game.game_id)); }
      catch (e) { showToast(e.message); }
      finally { setBusy(false); }
    }, 400);
    return () => clearTimeout(t);
  }, [game]);   // eslint-disable-line react-hooks/exhaustive-deps

  if (!game) {
    return (
      <div className="setup">
        <h1>gomoku-rl</h1>
        <p>五子棋强化学习实战 · M0</p>
        {["black", "white"].map((side) => (
          <label key={side}>
            {side === "black" ? "黑方" : "白方"}：
            <select value={config[side]}
                    onChange={(e) => setConfig({ ...config, [side]: e.target.value })}>
              {OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
        ))}
        <div>
          <button onClick={start} disabled={busy}>开始对局</button>
        </div>
        {toast && <div className="toast">{toast}</div>}
      </div>
    );
  }

  const currentKind = game.current_player === 1 ? config.black : config.white;
  const humanTurn = currentKind === "human";

  return (
    <div className="game">
      <div className="status">
        {game.game_over
          ? (game.winner === 3
              ? "平局"
              : `胜者：${labelOf(game.winner === 1 ? config.black : config.white)}`)
          : `轮到：${labelOf(currentKind)}${busy && !humanTurn ? "（思考中…）" : ""}`}
        <span>第 {game.moves.length} 手</span>
        <button onClick={() => setGame(null)}>重新开局</button>
      </div>
      <Board grid={game.board} lastMove={game.last_move} onPlay={play}
             locked={busy || game.game_over || !humanTurn} />
      <div className="toolbar">
        {PENDING.map(([label, milestone]) => (
          <button key={label} className="pending"
                  onClick={() => showToast(`${label}：后续开发（${milestone}）`)}>
            {label}
          </button>
        ))}
      </div>
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
