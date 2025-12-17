
import type { Piece } from '../../types';
import { Piece3D } from './Piece3D';
import type { ThreeEvent } from '@react-three/fiber';

interface Hand3DProps {
    player: 'orange' | 'blue';
    pieces: Piece[];
    onPieceClick: (piece: Piece) => void;
    selectedPiece: Piece | null;
    isActive: boolean;
    isMobile: boolean;
    onDrop: (row: number, col: number) => void;
}

export function Hand3D({ player, pieces, onPieceClick, selectedPiece, isActive, isMobile, onDrop }: Hand3DProps) {
    // Position hands on opposite sides of the board
    // Board is roughly -3.5 to +3.5. Let's place hands at z = +/- 6
    // Or maybe x = +/- 6.
    // Let's do Side by Side for landscape? Or Top/Bottom for standard board game view?
    // Let's do Top (Blue) and Bottom (Orange).

    // Mobile: Top/Bottom
    // Desktop: Top/Bottom but closer (z=6.5) -> User wants even closer.
    // Board edge is approx +/- 5.0.
    // Let's try 6.0 for both.

    const position: [number, number, number] = isMobile
        ? (player === 'orange' ? [0, 0, 6.0] : [0, 0, -6.0])
        : (player === 'orange' ? [0, 0, 6.0] : [0, 0, -6.0]);

    const rotation: [number, number, number] = player === 'blue'
        ? [0, Math.PI, 0] // Face the board
        : [0, 0, 0];

    // For mobile top-down, maybe we want to rotate them to face the camera?
    // Actually current rotation is fine if we are looking from top.

    return (
        <group position={position} rotation={rotation}>

            {pieces.map((piece, idx) => {
                const spacing = isMobile ? 1.3 : 1.5; // Increased mobile spacing further
                const startX = -((pieces.length - 1) * spacing) / 2;
                const x = startX + idx * spacing;

                const isSelected = selectedPiece?.id === piece.id;

                return (
                    <Piece3D
                        key={piece.id}
                        piece={piece}
                        position={[x, -0.5, 0]}
                        isSelected={isSelected}
                        onClick={(e: ThreeEvent<MouseEvent>) => {
                            e.stopPropagation();
                            // Only allow clicking own pieces if active turn (handled by App logic primarily, but visual feedback helps)
                            if (isActive) {
                                onPieceClick(piece);
                            }
                        }}
                        onDrop={onDrop}
                    />
                )
            })}
        </group>
    );
}

