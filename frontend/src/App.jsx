import Board from "./Board";

const EMPTY_GRID = Array.from({ length: 15 }, () => Array(15).fill(0));

export default function App() {
  return <Board grid={EMPTY_GRID} lastMove={null} onPlay={() => {}} locked />;
}
