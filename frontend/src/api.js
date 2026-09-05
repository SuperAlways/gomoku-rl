async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function get(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  newGame: (black, white, size = 15) => post("/api/new", { black, white, size }),
  move: (gameId, row, col) => post("/api/move", { game_id: gameId, row, col }),
  aiMove: (gameId) => post("/api/ai-move", { game_id: gameId }),
  undo: (gameId) => post("/api/undo", { game_id: gameId }),
  games: () => get("/api/games"),
  record: (gameId) => get(`/api/games/${gameId}/record`),
  downloadSgf: (gameId) => { window.location.assign(`/api/games/${gameId}/sgf`); },
};
