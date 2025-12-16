
import type { Piece } from '../../types';
import { Piece3D } from './Piece3D';
import type { ThreeEvent } from '@react-three/fiber';

interface Hand3DProps {
    player: 'orange' | 'blue';
    pieces: Piece[];
    onPieceClick: (piece: Piece) => void;
    selectedPiece: Piece | null;
    isActive: boolean;
}

export function Hand3D({ player, pieces, onPieceClick, selectedPiece, isActive }: Hand3DProps) {
    // Position hands on opposite sides of the board
    // Board is roughly -3.5 to +3.5. Let's place hands at z = +/- 6
    // Or maybe x = +/- 6.
    // Let's do Side by Side for landscape? Or Top/Bottom for standard board game view?
    // Let's do Top (Blue) and Bottom (Orange).

    const position: [number, number, number] = player === 'orange'
        ? [0, 0, 6] // Bottom
        : [0, 0, -6]; // Top

    const rotation: [number, number, number] = player === 'blue'
        ? [0, Math.PI, 0] // Face the board
        : [0, 0, 0];

    // Group pieces by size to organize them nicely? 
    // Or just display them in a line.
    // Let's organize by size: Small, Medium, Large.

    return (
        <group position={position} rotation={rotation}>
            {/* Base/Tray for hand */}
            <mesh receiveShadow position={[0, -0.4, 0]}>
                <boxGeometry args={[10, 0.5, 2]} />
                <meshStandardMaterial color={isActive ? "#334155" : "#1e293b"} />
            </mesh>

            {/* Label? Maybe UI overlay is better for names, but 3D text is cool */}
            {/* 
       <Text 
        position={[0, 0.5, 1.2]} 
        color={player === 'orange' ? "orange" : "skyblue"}
        fontSize={0.5}
        anchorX="center"
        anchorY="middle"
       >
        {player.toUpperCase()}
       </Text>
       */}

            {pieces.map((piece, idx) => {
                // Calculate slot position
                // We have up to 6 pieces (2 small, 2 med, 2 large) per player usually?
                // Actually createInitialHand gives: 2 Large, 2 Medium, 2 Small = 6 pieces.
                // Let's distribute them: Large(L, R), Med(L, R), Small(L, R)
                // Or just -3, -2, -1, 1, 2, 3

                const spacing = 1.3;
                const startX = -((pieces.length - 1) * spacing) / 2;
                const x = startX + idx * spacing;

                const isSelected = selectedPiece?.id === piece.id;

                return (
                    <Piece3D
                        key={piece.id}
                        piece={piece}
                        position={[x, 0, 0]}
                        isSelected={isSelected}
                        onClick={(e: ThreeEvent<MouseEvent>) => {
                            e.stopPropagation();
                            // Only allow clicking own pieces if active turn (handled by App logic primarily, but visual feedback helps)
                            if (isActive) {
                                onPieceClick(piece);
                            }
                        }}
                    />
                )
            })}
        </group>
    );
}

