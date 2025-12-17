import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import type { BoardState, Piece, Player, SelectedPiece } from '../../types';
import { Board3D } from './Board3D';
import { Hand3D } from './Hand3D';
import { Suspense } from 'react';
import * as THREE from 'three';

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
    isMobile: boolean;
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
    isMobile,
}: SceneProps) {

    return (
        <Canvas
            shadows
            dpr={[1, 2]}
            className="w-full h-full absolute inset-0"
            onContextMenu={(e) => e.preventDefault()}
        >
            <Suspense fallback={null}>
                <PerspectiveCamera
                    makeDefault
                    position={isMobile ? [0, 15, 0] : [0, 8, 12]}
                    fov={50}
                />
                <OrbitControls
                    minPolarAngle={0}
                    maxPolarAngle={Math.PI / 2.1}
                    maxDistance={20}
                    minDistance={5}
                    enablePan={false}
                    mouseButtons={{
                        LEFT: THREE.MOUSE.PAN, // Pan is disabled; frees left-drag for selecting/moving pieces
                        MIDDLE: THREE.MOUSE.DOLLY,
                        RIGHT: THREE.MOUSE.ROTATE,
                    }}
                />

                {/* Lighting - High Key Studio */}
                <ambientLight intensity={1.2} />
                <spotLight
                    position={[5, 15, 5]}
                    angle={0.4}
                    penumbra={0.5}
                    intensity={1.0}
                    castShadow
                    shadow-bias={-0.0001}
                />
                <pointLight position={[-10, 5, -10]} intensity={0.5} />

                {/* Studio Background */}
                <color attach="background" args={['#ffffff']} />

                {/* Floor Shadows only */}
                <group position={[0, -1.51, 0]}> {/* At bottom of bars (-1.5) with tiny offset to prevent z-fighting if any */}
                    <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
                        <planeGeometry args={[100, 100]} />
                        <shadowMaterial opacity={0.1} />
                    </mesh>
                </group>

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
                        isMobile={isMobile}
                        onDrop={onCellClick}
                    />

                    <Hand3D
                        player="blue"
                        pieces={blueHand}
                        isActive={turn === 'blue' && !winner}
                        selectedPiece={selectedPiece?.piece || null}
                        onPieceClick={(p) => onPieceClick(p, 'hand')}
                        isMobile={isMobile}
                        onDrop={onCellClick}
                    />
                </group>
            </Suspense>
        </Canvas>
    );
}
