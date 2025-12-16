import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stars } from '@react-three/drei';
import type { BoardState, Piece, Player, SelectedPiece } from '../../types';
import { Board3D } from './Board3D';
import { Hand3D } from './Hand3D';
import { Suspense } from 'react';

interface SceneProps {
    board: BoardState;
    turn: Player;
    winner: Player | 'draw' | null;
    orangeHand: Piece[];
    blueHand: Piece[];
    selectedPiece: SelectedPiece | null;
    onPieceClick: (piece: Piece, from: 'hand' | { row: number; col: number }) => void;
    onCellClick: (row: number, col: number) => void;
    isValidMove: (row: number, col: number) => boolean;
}

export function Scene({
    board,
    turn,
    winner,
    orangeHand,
    blueHand,
    selectedPiece,
    onPieceClick,
    onCellClick,
    isValidMove,
}: SceneProps) {

    return (
        <Canvas shadows dpr={[1, 2]} className="w-full h-full absolute inset-0">
            <Suspense fallback={null}>
                <PerspectiveCamera makeDefault position={[0, 8, 12]} fov={50} />
                <OrbitControls
                    minPolarAngle={0}
                    maxPolarAngle={Math.PI / 2.1}
                    maxDistance={20}
                    minDistance={5}
                />

                {/* Lighting */}
                <ambientLight intensity={0.5} />
                <spotLight
                    position={[10, 10, 10]}
                    angle={0.15}
                    penumbra={1}
                    intensity={1}
                    castShadow
                    shadow-mapSize={[2048, 2048]}
                />
                <pointLight position={[-10, -10, -10]} intensity={0.5} />

                {/* Environment / Background */}
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
                <color attach="background" args={['#0f172a']} /> {/* slate-900 */}

                {/* Floor Reflections */}
                {/* <ContactShadows resolution={1024} scale={20} blur={2} opacity={0.5} far={10} color="#000000" /> */}

                <group position={[0, -1, 0]}>
                    <Board3D
                        board={board}
                        onCellClick={onCellClick}
                        selectedPiece={selectedPiece}
                        isValidMove={isValidMove}
                    />

                    <Hand3D
                        player="orange"
                        pieces={orangeHand}
                        isActive={turn === 'orange' && !winner}
                        selectedPiece={selectedPiece?.piece || null}
                        onPieceClick={(p) => onPieceClick(p, 'hand')}
                    />

                    <Hand3D
                        player="blue"
                        pieces={blueHand}
                        isActive={turn === 'blue' && !winner}
                        selectedPiece={selectedPiece?.piece || null}
                        onPieceClick={(p) => onPieceClick(p, 'hand')}
                    />
                </group>
            </Suspense>
        </Canvas>
    );
}
