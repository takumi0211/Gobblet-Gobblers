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
            {/* Bar Grid Structure */}
            <group position={[0, -0.25, 0]}>
                {/* Horizontal Bars (Cyan/Blue) - along Z axis? No, cutting across Z axis, so acting as rows dividers */}
                {/* Positions: Between row 0&1 (z=-1.75) and 1&2 (z=1.75) */}

                {/* Bar 1 (Top) */}
                <mesh receiveShadow castShadow position={[0, 0, -1.75]}>
                    <boxGeometry args={[11, 0.5, 0.5]} />
                    <meshStandardMaterial color="#0ea5e9" roughness={0.4} /> {/* sky-500 */}
                </mesh>

                {/* Bar 2 (Bottom) */}
                <mesh receiveShadow castShadow position={[0, 0, 1.75]}>
                    <boxGeometry args={[11, 0.5, 0.5]} />
                    <meshStandardMaterial color="#0ea5e9" roughness={0.4} /> {/* sky-500 */}
                </mesh>

                {/* Vertical Bars (Orange) - along X axis? No, cutting across X axis, so acting as col dividers */}
                {/* Positions: Between col 0&1 (x=-1.75) and 1&2 (x=1.75) */}

                {/* Bar 3 (Left) - Shifted Y slightly to interlock or stack? Let's stack them on top for now or cross them */}
                {/* Reference image: They seem to be on same level or interlocking. Simple cross is fine. */}
                <mesh receiveShadow castShadow position={[-1.75, 0, 0]}>
                    <boxGeometry args={[0.5, 0.5, 11]} />
                    <meshStandardMaterial color="#f97316" roughness={0.4} /> {/* orange-500 */}
                </mesh>

                {/* Bar 4 (Right) */}
                <mesh receiveShadow castShadow position={[1.75, 0, 0]}>
                    <boxGeometry args={[0.5, 0.5, 11]} />
                    <meshStandardMaterial color="#f97316" roughness={0.4} /> {/* orange-500 */}
                </mesh>
            </group>

            {/* Cells */}
            {board.map((row, rIdx) =>
                row.map((cellState, cIdx) => {
                    const x = (cIdx - 1) * cellSpacing;
                    const z = (rIdx - 1) * cellSpacing;
                    const isValid = isValidMove(rIdx, cIdx);

                    return (
                        <group key={`${rIdx}-${cIdx}`} position={[x, 0, z]}>
                            {/* Cell Hit Area / Visual - Only visible if valid or debug */}
                            <mesh
                                rotation={[-Math.PI / 2, 0, 0]}
                                position={[0, -0.2, 0]} // Slightly below pieces
                                onClick={() => {
                                    onCellClick(rIdx, cIdx);
                                }}
                            >
                                <planeGeometry args={[3, 3]} />
                                <meshStandardMaterial
                                    color={isValid ? "#86efac" : "white"}
                                    transparent
                                    opacity={isValid ? 0.3 : 0.0} // Invisible unless valid
                                    side={2} // Double side
                                />
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
                                        position={[0, -0.5, 0]} // On floor (bottom of grid)
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
