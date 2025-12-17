import { useEffect, useState } from 'react';
import { useIsMobile } from './hooks/useIsMobile';
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
  const isMobile = useIsMobile();
  const [board, setBoard] = useState<BoardState>(createInitialBoard());
  const [turn, setTurn] = useState<Player>('orange');
  const [winner, setWinner] = useState<Player | 'draw' | null>(null);
  const [orangeHand, setOrangeHand] = useState<Piece[]>(createInitialHand('orange'));
  const [blueHand, setBlueHand] = useState<Piece[]>(createInitialHand('blue'));
  const [selectedPiece, setSelectedPiece] = useState<SelectedPiece | null>(null);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedPiece(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const handlePieceClick = (piece: Piece, from: 'hand' | { row: number; col: number }) => {
    if (winner) return;
    if (piece.player !== turn) return;

    // Toggle off if clicking the currently selected piece
    if (selectedPiece?.piece.id === piece.id) {
      setSelectedPiece(null);
      return;
    }

    // If clicking a piece on board, make sure it's the top one
    if (from !== 'hand') {
      const cell = board[from.row][from.col];
      const topPiece = getTopPiece(cell);
      if (topPiece?.id !== piece.id) return; // Can only move top piece
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
        onCellClick={(r, c) => {
          // If clicking a cell with a piece of current player, select it (if no piece selected or switching selection)
          const cell = board[r][c];
          const topPiece = getTopPiece(cell);
          if (!selectedPiece && topPiece && topPiece.player === turn) {
            handlePieceClick(topPiece, { row: r, col: c });
          } else {
            handleCellClick(r, c);
          }
        }}
        isValidMove={(r, c) => {
          if (!selectedPiece) return false;
          const fromRow = selectedPiece.from !== 'hand' ? selectedPiece.from.row : undefined;
          const fromCol = selectedPiece.from !== 'hand' ? selectedPiece.from.col : undefined;
          return isValidMove(r, c, selectedPiece.piece, board, fromRow, fromCol);
        }}
        isMobile={isMobile}
      />

      {/* UI Overlay */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none flex flex-col justify-between p-8">

        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-blue-400 drop-shadow-lg transition-all duration-300">
            Gobblet Gobblers
          </h1>
          <div className="mt-4 text-xl md:text-3xl font-bold font-mono tracking-wider drop-shadow-md">
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
            className="px-8 py-3 bg-slate-800 hover:bg-slate-700 text-white border border-slate-600 rounded-full font-bold text-lg transition-all transform hover:scale-105 active:scale-95 shadow-lg"
          >
            Reset Game
          </button>
        </div>
      </div>

      {/* Instructions / Controls Hint */}
      <div className="absolute bottom-4 right-4 text-slate-400 text-xs pointer-events-none">
        Left Click to Select/Move • Right Drag to Rotate • Wheel to Zoom • Esc to Cancel
      </div>
    </div>
  );
}

export default App;
