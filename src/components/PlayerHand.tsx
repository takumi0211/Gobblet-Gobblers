import React from 'react';
import type { Piece as PieceType, Player } from '../types';
import { Piece } from './Piece';

interface PlayerHandProps {
    player: Player;
    pieces: PieceType[];
    onPieceClick: (piece: PieceType) => void;
    selectedPiece: PieceType | null;
    isActive: boolean;
}

export const PlayerHand: React.FC<PlayerHandProps> = ({ player, pieces, onPieceClick, selectedPiece, isActive }) => {
    return (
        <div className={`
      flex flex-col items-center gap-4 p-4 rounded-xl transition-colors
      ${isActive ? 'bg-opacity-20 bg-gray-500' : 'opacity-50'}
    `}>
            <h3 className={`text-xl font-bold capitalize ${player === 'orange' ? 'text-orange-500' : 'text-blue-500'}`}>
                {player} Hand
            </h3>
            <div className="flex gap-2 flex-wrap justify-center min-h-[80px]">
                {pieces.map((piece) => (
                    <Piece
                        key={piece.id}
                        piece={piece}
                        onClick={() => isActive && onPieceClick(piece)}
                        isSelected={selectedPiece?.id === piece.id}
                    />
                ))}
                {pieces.length === 0 && (
                    <span className="text-gray-400 text-sm italic">No pieces left</span>
                )}
            </div>
        </div>
    );
};
