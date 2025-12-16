import React from 'react';
import type { Piece as PieceType } from '../types';
import { Piece } from './Piece';

interface CellProps {
    pieces: PieceType[];
    row: number;
    col: number;
    onClick: () => void;
    isValidTarget?: boolean;
}

export const Cell: React.FC<CellProps> = ({ pieces, onClick, isValidTarget }) => {
    const topPiece = pieces.length > 0 ? pieces[pieces.length - 1] : null;

    return (
        <div
            onClick={onClick}
            className={`
        w-24 h-24 border-2 border-gray-300 flex items-center justify-center relative bg-white
        ${isValidTarget ? 'bg-green-100 hover:bg-green-200 cursor-pointer' : ''}
      `}
        >
            {/* Render grid lines visual aid if needed, but border handles it */}

            {topPiece && (
                <div className="z-10">
                    <Piece piece={topPiece} />
                </div>
            )}

            {/* Show shadow of pieces underneath? Maybe too complex for now. */}
        </div>
    );
};
