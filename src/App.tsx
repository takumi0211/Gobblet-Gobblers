import { useState } from 'react';
import { Board } from './components/Board';
import { PlayerHand } from './components/PlayerHand';
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
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-blue-400">
        Gobblet Gobblers
      </h1>

      <div className="flex flex-col md:flex-row gap-8 items-center">
        <PlayerHand
          player="orange"
          pieces={orangeHand}
          onPieceClick={(p) => handlePieceClick(p, 'hand')}
          selectedPiece={selectedPiece?.piece || null}
          isActive={turn === 'orange' && !winner}
        />

        <div className="flex flex-col items-center gap-4">
          <div className="text-2xl font-semibold mb-2">
            {winner ? (
              <span className={winner === 'orange' ? 'text-orange-500' : 'text-blue-500'}>
                {winner.toUpperCase()} WINS!
              </span>
            ) : (
              <span className={turn === 'orange' ? 'text-orange-500' : 'text-blue-500'}>
                {turn.toUpperCase()}'s Turn
              </span>
            )}
          </div>

          <Board
            board={board}
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
            selectedPiece={selectedPiece}
            isValidMove={(r, c) => {
              if (!selectedPiece) return false;
              const fromRow = selectedPiece.from !== 'hand' ? selectedPiece.from.row : undefined;
              const fromCol = selectedPiece.from !== 'hand' ? selectedPiece.from.col : undefined;
              return isValidMove(r, c, selectedPiece.piece, board, fromRow, fromCol);
            }}
          />

          <button
            onClick={resetGame}
            className="mt-4 px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold transition-colors"
          >
            Reset Game
          </button>
        </div>

        <PlayerHand
          player="blue"
          pieces={blueHand}
          onPieceClick={(p) => handlePieceClick(p, 'hand')}
          selectedPiece={selectedPiece?.piece || null}
          isActive={turn === 'blue' && !winner}
        />
      </div>
    </div>
  );
}

export default App;
