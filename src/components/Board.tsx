import React from 'react';
import type { BoardState, SelectedPiece } from '../types';
import { Cell } from './Cell';

interface BoardProps {
    board: BoardState;
    onCellClick: (row: number, col: number) => void;
    selectedPiece: SelectedPiece | null;
    isValidMove: (row: number, col: number) => boolean;
}

export const Board: React.FC<BoardProps> = ({ board, onCellClick, selectedPiece, isValidMove }) => {
    return (
        <div className="grid grid-cols-3 gap-2 bg-gray-800 p-2 rounded-lg shadow-xl">
            {board.map((row, rowIndex) => (
                <React.Fragment key={rowIndex}>
                    {row.map((cell, colIndex) => (
                        <Cell
                            key={`${rowIndex}-${colIndex}`}
                            row={rowIndex}
                            col={colIndex}
                            pieces={cell}
                            onClick={() => onCellClick(rowIndex, colIndex)}
                            isValidTarget={selectedPiece ? isValidMove(rowIndex, colIndex) : false}
                        />
                    ))}
                </React.Fragment>
            ))}
        </div>
    );
};
