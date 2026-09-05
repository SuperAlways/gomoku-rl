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

export const api = {
  newGame: (black, white) => post("/api/new", { black, white }),
  move: (gameId, row, col) => post("/api/move", { game_id: gameId, row, col }),
  aiMove: (gameId) => post("/api/ai-move", { game_id: gameId }),
  games: () => fetch("/api/games").then((r) => r.json()),
};
