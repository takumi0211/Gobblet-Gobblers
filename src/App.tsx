import { useState } from 'react';
import { Scene } from './components/3d/Scene';
import type { BoardState, Piece, Player, SelectedPiece } from './types';
import {
  checkWin,
  cloneBoard,
  createInitialBoard,
  createInitialHand,
  getTopPiece,
  isValidMove,
} from './utils/gameLogic';

function App() {
  const [board, setBoard] = useState<BoardState>(createInitialBoard());
  const [turn, setTurn] = useState<Player>('orange');
  const [winner, setWinner] = useState<Player | 'draw' | null>(null);
  const [orangeHand, setOrangeHand] = useState<Piece[]>(createInitialHand('orange'));
  const [blueHand, setBlueHand] = useState<Piece[]>(createInitialHand('blue'));
  const [selectedPiece, setSelectedPiece] = useState<SelectedPiece | null>(null);

  const handlePieceClick = (piece: Piece, from: 'hand' | { row: number; col: number }) => {
    if (winner) return;
    if (piece.player !== turn) return;

    // If you already "touched" a piece on the board, you must move it (can't cancel/switch).
    if (selectedPiece && selectedPiece.from !== 'hand') {
      if (selectedPiece.piece.id === piece.id) return;
      return;
    }

    // If clicking the same piece from hand, deselect
    if (selectedPiece?.piece.id === piece.id && selectedPiece.from === 'hand' && from === 'hand') {
      setSelectedPiece(null);
      return;
    }

    // If clicking a piece on board, make sure it's the top one
    if (from !== 'hand') {
      const cell = board[from.row][from.col];
      const topPiece = getTopPiece(cell);
      if (topPiece?.id !== piece.id) return; // Can only move top piece

      // If the touched piece has no legal destination, don't allow selecting it (prevents a dead lock).
      const hasAnyValidMove = board.some((r, rIdx) =>
        r.some((_c, cIdx) => isValidMove(rIdx, cIdx, piece, board, from.row, from.col))
      );
      if (!hasAnyValidMove) return;
    }

    setSelectedPiece({ piece, from });
  };

  const handleCellClick = (row: number, col: number) => {
    if (winner || !selectedPiece) return;

    // Check if move is valid
    const fromRow = selectedPiece.from !== 'hand' ? selectedPiece.from.row : undefined;
    const fromCol = selectedPiece.from !== 'hand' ? selectedPiece.from.col : undefined;

    if (!isValidMove(row, col, selectedPiece.piece, board, fromRow, fromCol)) {
      // If invalid move, maybe they meant to select the piece at this cell?
      const cell = board[row][col];
      const topPiece = getTopPiece(cell);
      if (topPiece && topPiece.player === turn) {
        handlePieceClick(topPiece, { row, col });
      }
      return;
    }

    // Execute move
    const newBoard = cloneBoard(board);

    // Remove from source
    if (selectedPiece.from === 'hand') {
      if (turn === 'orange') {
        setOrangeHand(prev => prev.filter(p => p.id !== selectedPiece.piece.id));
      } else {
        setBlueHand(prev => prev.filter(p => p.id !== selectedPiece.piece.id));
      }
    } else {
      const { row: fRow, col: fCol } = selectedPiece.from;
      newBoard[fRow][fCol].pop(); // Remove top piece

      // Rule: if lifting reveals a 3-in-a-row, the game ends immediately.
      const winAfterLift = checkWin(newBoard);
      if (winAfterLift) {
        setBoard(newBoard);
        setSelectedPiece(null);
        setWinner(winAfterLift);
        return;
      }
    }

    // Add to destination
    newBoard[row][col].push(selectedPiece.piece);
    setBoard(newBoard);
    setSelectedPiece(null);

    // Check win
    const win = checkWin(newBoard);
    if (win) {
      setWinner(win);
    } else {
      setTurn(turn === 'orange' ? 'blue' : 'orange');
    }
  };

  const resetGame = () => {
    setBoard(createInitialBoard());
    setTurn('orange');
    setWinner(null);
    setOrangeHand(createInitialHand('orange'));
    setBlueHand(createInitialHand('blue'));
    setSelectedPiece(null);
  };

  return (
    <div className="w-full h-screen bg-gray-900 overflow-hidden relative selection:bg-none">
      {/* 3D Scene */}
      <Scene
        board={board}
        turn={turn}
        winner={winner}
        orangeHand={orangeHand}
        blueHand={blueHand}
        selectedPiece={selectedPiece}
        onPieceClick={handlePieceClick}
        onCellClick={handleCellClick}
        isValidMove={(r, c) => {
          if (!selectedPiece) return false;
          const fromRow = selectedPiece.from !== 'hand' ? selectedPiece.from.row : undefined;
          const fromCol = selectedPiece.from !== 'hand' ? selectedPiece.from.col : undefined;
          return isValidMove(r, c, selectedPiece.piece, board, fromRow, fromCol);
        }}
      />

      {/* UI Overlay */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none flex flex-col justify-between p-8">

        {/* Header */}
        <div className="text-center">
          <h1 className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-blue-400 drop-shadow-lg"
            style={{ filter: 'drop-shadow(0 0 10px rgba(0,0,0,0.5))' }}>
            Gobblet Gobblers
          </h1>
          <div className="mt-4 text-3xl font-bold font-mono tracking-wider drop-shadow-md">
            {winner ? (
              <span className={`${winner === 'orange' ? 'text-orange-500' : 'text-blue-500'} animate-pulse`}>
                {winner.toUpperCase()} WINS!
              </span>
            ) : (
              <span className={turn === 'orange' ? 'text-orange-500' : 'text-blue-500'}>
                {turn === 'orange' ? 'ORANGE' : 'BLUE'}'s TURN
              </span>
            )}
          </div>
        </div>

        {/* Footer / Controls */}
        <div className="flex justify-center pb-8 pointer-events-auto">
          <button
            onClick={resetGame}
            className="px-8 py-3 bg-white/10 backdrop-blur-md hover:bg-white/20 border border-white/20 rounded-full text-white font-bold text-lg transition-all transform hover:scale-105 active:scale-95 shadow-lg"
          >
            Reset Game
          </button>
        </div>
      </div>

      {/* Instructions / Controls Hint */}
      <div className="absolute bottom-4 right-4 text-white/30 text-xs pointer-events-none">
        Left Click to Select/Move • Right Click to Rotate • Wheel to Zoom
      </div>
    </div>
  );
}

export default App;

