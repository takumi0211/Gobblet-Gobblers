import type { BoardState, SelectedPiece } from '../../types';
import { Piece3D } from './Piece3D';
import type { ThreeEvent } from '@react-three/fiber';

interface Board3DProps {
    board: BoardState;
    onCellClick: (row: number, col: number) => void;
    selectedPiece: SelectedPiece | null;
    isValidMove: (row: number, col: number) => boolean;
}

export function Board3D({ board, onCellClick, selectedPiece, isValidMove }: Board3DProps) {
    const cellSpacing = 3.5; // Distance between cell centers

    return (
        <group>
            {/* Base Platform */}
            <mesh receiveShadow position={[0, -0.5, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                <boxGeometry args={[12, 12, 1]} />
                <meshStandardMaterial color="#1e293b" /> {/* slate-800 */}
            </mesh>

            {/* Grid Lines */}
            <gridHelper args={[10.5, 3, "#94a3b8", "#334155"]} position={[0, 0.01, 0]} />

            {/* Cells */}
            {board.map((row, rIdx) =>
                row.map((cellState, cIdx) => {
                    const x = (cIdx - 1) * cellSpacing;
                    const z = (rIdx - 1) * cellSpacing;
                    const isValid = isValidMove(rIdx, cIdx);

                    // const topPiece = getTopPiece(cellState); // Removed unused variable

                    return (
                        <group key={`${rIdx}-${cIdx}`} position={[x, 0, z]}>
                            {/* Cell Hit Area / Visual */}
                            <mesh
                                rotation={[-Math.PI / 2, 0, 0]}
                                position={[0, 0.02, 0]}
                                onClick={() => {
                                    onCellClick(rIdx, cIdx);
                                }}
                            >
                                <planeGeometry args={[3, 3]} />
                                <meshStandardMaterial
                                    color={isValid ? "#34d399" : "#1e293b"} // Green if valid, else slate
                                    transparent
                                    opacity={isValid ? 0.3 : 0}
                                />
                            </mesh>

                            {/* Cell Border/Marker */}
                            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
                                <ringGeometry args={[1.4, 1.5, 32]} />
                                <meshStandardMaterial color="#334155" />
                            </mesh>

                            {/* Render Pieces in this cell */}
                            {cellState.map((piece, pIdx) => {
                                // Only render visible pieces? Ideally yes just the top one or stack slightly?
                                // In real game, bigger ones cover smaller ones. 
                                // Implementing true "Gobbling" visually:
                                // Render all, but smaller ones are hidden inside bigger ones automatically by geometry if sizes are correct.
                                // But to be safe and efficient, we can just render all at same pos, z-fighting might be issue if exact same size.
                                // Since sizes differ, geometry handles it.

                                const isTop = pIdx === cellState.length - 1;
                                const isSelected = selectedPiece?.piece.id === piece.id;

                                return (
                                    <Piece3D
                                        key={piece.id}
                                        piece={piece}
                                        position={[0, 0, 0]} // Relative to cell group
                                        isSelected={isSelected}
                                        isTop={isTop}
                                        onClick={(e: ThreeEvent<MouseEvent>) => {
                                            e.stopPropagation();
                                            // Click handled by Board interaction generally, but we can pass it up
                                            onCellClick(rIdx, cIdx);
                                        }}
                                    />
                                )
                            })}
                        </group>
                    );
                })
            )}
        </group>
    );
}
