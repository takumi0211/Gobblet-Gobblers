import React from 'react';
import type { Piece as PieceType } from '../types';

interface PieceProps {
    piece: PieceType;
    onClick?: () => void;
    isSelected?: boolean;
}

export const Piece: React.FC<PieceProps> = ({ piece, onClick, isSelected }) => {
    const sizeClasses = {
        small: 'w-8 h-8',
        medium: 'w-12 h-12',
        large: 'w-16 h-16',
    };

    const colorClasses = {
        orange: 'bg-orange-500 border-orange-700',
        blue: 'bg-blue-500 border-blue-700',
    };

    return (
        <div
            onClick={onClick}
            className={`
        rounded-full border-4 flex items-center justify-center shadow-lg transition-transform cursor-pointer
        ${sizeClasses[piece.size]}
        ${colorClasses[piece.player]}
        ${isSelected ? 'ring-4 ring-yellow-400 scale-110' : ''}
      `}
        >
            {/* Inner circle for decoration */}
            <div className="w-1/2 h-1/2 rounded-full bg-white opacity-30" />
        </div>
    );
};
