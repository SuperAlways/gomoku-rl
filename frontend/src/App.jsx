import { useEffect, useRef, useState } from "react";
import Board from "./Board";
import Replay from "./Replay";
import { api } from "./api";

const OPTIONS = [
  { value: "human", label: "玩家" },
  { value: "minimax-easy", label: "Minimax·简单" },
  { value: "minimax-medium", label: "Minimax·中等" },
  { value: "minimax-hard", label: "Minimax·困难" },
];
const PENDING = [
  ["RL 对战", "M2+"], ["训练面板", "M4+"],
];

function labelOf(kind) {
  return OPTIONS.find((o) => o.value === kind)?.label
    ?? ({ qtable: "九宫格·Q表" }[kind] ?? kind);
}

export default function App() {
  const [view, setView] = useState("config");   // config | game | history | replay
  const [config, setConfig] = useState({ black: "human", white: "minimax-hard" });
  const [mode, setMode] = useState("standard");   // standard | qtable3
  const [game, setGame] = useState(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");
  const [showRematch, setShowRematch] = useState(false);
  const [historyList, setHistoryList] = useState(null);
  const [replayRecord, setReplayRecord] = useState(null);
  const toastTimer = useRef(null);

  const showToast = (msg) => {
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 2200);
  };

  const start = async () => {
    setBusy(true);
    try {
      if (mode === "qtable3") {
        setConfig({ black: "qtable", white: "human" });
        setGame(await api.newGame("qtable", "human", 3));
      } else {
        setGame(await api.newGame(config.black, config.white, 15));
      }
      setView("game");
    } catch (e) { showToast(e.message); }
    finally { setBusy(false); }
  };

  const play = async (row, col) => {
    if (busy) return;
    setBusy(true);
    try { setGame(await api.move(game.game_id, row, col)); }
    catch (e) { showToast(e.message); }
    finally { setBusy(false); }
  };

  const undo = async () => {
    if (busy) return;
    setBusy(true);
    try { setGame(await api.undo(game.game_id)); }
    catch (e) { showToast(e.message); }
    finally { setBusy(false); }
  };

  // AI 回合：延迟请求一手；机 vs 机时随状态更新自动连续走
  useEffect(() => {
    if (view !== "game" || !game || game.game_over) return;
    const side = game.current_player === 1 ? config.black : config.white;
    if (side === "human") return;
    const t = setTimeout(async () => {
      setBusy(true);
      try { setGame(await api.aiMove(game.game_id)); }
      catch (e) { showToast(e.message); }
      finally { setBusy(false); }
    }, 400);
    return () => clearTimeout(t);
  }, [game, view]);   // eslint-disable-line react-hooks/exhaustive-deps

  // 终局 30 秒后提示重开（含平局）
  useEffect(() => {
    if (!game?.game_over) { setShowRematch(false); return; }
    const t = setTimeout(() => setShowRematch(true), 30_000);
    return () => clearTimeout(t);
  }, [game]);

  // 进入历史视图时拉取列表
  useEffect(() => {
    if (view !== "history") return;
    api.games().then(setHistoryList).catch((e) => {
      setHistoryList([]);
      showToast(e.message);
    });
  }, [view]);

  const backToConfig = () => {
    setGame(null);
    setView("config");
  };

  const goHistory = () => {
    setGame(null);
    setHistoryList(null);
    setView("history");
  };

  const openReplay = async (gid) => {
    try {
      setReplayRecord(await api.record(gid));
      setView("replay");
    } catch (e) { showToast(e.message); }
  };

  if (view === "history") {
    return (
      <div className="setup">
        <h1>历史对局</h1>
        {!historyList ? <p>加载中…</p>
          : historyList.length === 0 ? <p>还没有对局</p> : (
          <table className="history">
            <thead>
              <tr><th>时间</th><th>黑方</th><th>白方</th><th>结果</th><th>手数</th><th></th></tr>
            </thead>
            <tbody>
              {historyList.map((g) => (
                <tr key={g.id}>
                  <td>{g.started_at}</td>
                  <td>{g.black_name}</td>
                  <td>{g.white_name}</td>
                  <td>{g.result}</td>
                  <td>{g.total_moves}</td>
                  <td>
                    <button onClick={() => openReplay(g.id)}>回放</button>
                    {" "}
                    <button onClick={() => api.downloadSgf(g.id)}>SGF</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div><button onClick={() => setView("config")}>返回</button></div>
        {toast && <div className="toast">{toast}</div>}
      </div>
    );
  }

  if (view === "replay") {
    return <Replay record={replayRecord} onBack={goHistory} />;
  }

  if (!game) {
    return (
      <div className="setup">
        <h1>gomoku-rl</h1>
        <p>五子棋强化学习实战 · M0</p>
        <div className="mode-row">
          <label>
            <input type="radio" name="mode" checked={mode === "standard"}
                   onChange={() => {
                     setMode("standard");
                     if (config.black === "qtable") {
                       setConfig({ black: "human", white: "minimax-hard" });
                     }
                   }} />
            标准 15×15
          </label>
          <label>
            <input type="radio" name="mode" checked={mode === "qtable3"}
                   onChange={() => setMode("qtable3")} />
            九宫格·Q表（3×3，E0）
          </label>
        </div>
        {mode === "qtable3" && <p>黑方：九宫格·Q表 AI　白方：玩家</p>}
        {mode === "standard" && ["black", "white"].map((side) => (
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
        <div>
          <button onClick={goHistory}>历史对局</button>
        </div>
        {toast && <div className="toast">{toast}</div>}
      </div>
    );
  }

  const currentKind = game.current_player === 1 ? config.black : config.white;
  const humanTurn = currentKind === "human";
  const bothHuman = config.black === "human" && config.white === "human";
  const canUndo = !game.game_over
    && (bothHuman ? game.moves.length >= 1 : humanTurn && game.moves.length >= 2);

  return (
    <div className="game">
      <div className="status">
        {game.game_over
          ? (game.winner === 3
              ? "平局"
              : `胜者：${labelOf(game.winner === 1 ? config.black : config.white)}`)
          : `轮到：${labelOf(currentKind)}${busy && !humanTurn ? "（思考中…）" : ""}`}
        <span>第 {game.moves.length} 手</span>
        <button onClick={backToConfig}>重新开局</button>
      </div>
      <Board grid={game.board} lastMove={game.last_move} winLine={game.win_line}
             onPlay={play} locked={busy || game.game_over || !humanTurn}
             size={config.black === "qtable" ? 3 : 15} />
      {game.game_over && showRematch && (
        <div className="rematch-bar">
          <span>再来一局？</span>
          <button onClick={backToConfig}>重开一局</button>
          <button onClick={() => setShowRematch(false)}>继续看</button>
        </div>
      )}
      <div className="toolbar">
        <button onClick={undo} disabled={busy || !canUndo}>悔棋</button>
        <button onClick={() => api.downloadSgf(game.game_id)}>棋谱下载</button>
        <button onClick={goHistory}>历史对局</button>
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
