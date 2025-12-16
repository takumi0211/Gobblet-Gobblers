
import { useRef, useMemo } from 'react';
import { useSpring, animated, config } from '@react-spring/three';
import * as THREE from 'three';
import type { ThreeEvent } from '@react-three/fiber';
import type { Piece } from '../../types';

interface Piece3DProps {
    piece: Piece;
    position: [number, number, number];
    isSelected?: boolean;
    onClick?: (e: ThreeEvent<MouseEvent>) => void;
    isTop?: boolean; // Only top pieces are interactive
}

export function Piece3D({ piece, position, isSelected, onClick, isTop = true }: Piece3DProps) {
    const meshRef = useRef<THREE.Group>(null);

    // Size configuration
    const scale = useMemo(() => {
        switch (piece.size) {
            case 'small': return 0.5;
            case 'medium': return 0.75;
            case 'large': return 1.0;
        }
    }, [piece.size]);

    // Color configuration
    const color = piece.player === 'orange' ? '#fb923c' : '#60a5fa'; // orange-400 : blue-400
    const emissive = isSelected ? '#ffffff' : '#000000';
    const emissiveIntensity = isSelected ? 0.5 : 0;

    // Animation for position and selection "hop"
    const { pos } = useSpring({
        pos: [position[0], position[1] + (isSelected ? 0.5 : 0), position[2]],
        config: config.wobbly,
    });

    // Eyes look slightly different for players
    const eyeColor = '#ffffff';
    const pupilColor = '#000000';

    return (
        <animated.group
            ref={meshRef}
            position={pos as any}
            scale={scale}
            onClick={(e) => {
                if (!isTop) return;
                e.stopPropagation();
                onClick?.(e);
            }}
        >
            {/* Helper to block raycasts for inner hidden pieces if needed, or just visual group */}

            {/* Main Body - Cylinder/Capsule like */}
            <mesh castShadow receiveShadow position={[0, 1, 0]}>
                <cylinderGeometry args={[0.6, 0.7, 2, 32]} />
                <meshStandardMaterial
                    color={color}
                    roughness={0.3}
                    metalness={0.1}
                    emissive={emissive}
                    emissiveIntensity={emissiveIntensity}
                />
            </mesh>

            {/* Rounded Top */}
            <mesh castShadow receiveShadow position={[0, 2, 0]}>
                <sphereGeometry args={[0.6, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
                <meshStandardMaterial
                    color={color}
                    roughness={0.3}
                    metalness={0.1}
                    emissive={emissive}
                    emissiveIntensity={emissiveIntensity}
                />
            </mesh>

            {/* Eyes - Left */}
            <group position={[-0.25, 1.6, 0.45]} rotation={[0, 0, 0]}>
                <mesh castShadow>
                    <sphereGeometry args={[0.15, 16, 16]} />
                    <meshStandardMaterial color={eyeColor} />
                </mesh>
                <mesh position={[0, 0, 0.12]}>
                    <sphereGeometry args={[0.06, 16, 16]} />
                    <meshStandardMaterial color={pupilColor} />
                </mesh>
            </group>

            {/* Eyes - Right */}
            <group position={[0.25, 1.6, 0.45]} rotation={[0, 0, 0]}>
                <mesh castShadow>
                    <sphereGeometry args={[0.15, 16, 16]} />
                    <meshStandardMaterial color={eyeColor} />
                </mesh>
                <mesh position={[0, 0, 0.12]}>
                    <sphereGeometry args={[0.06, 16, 16]} />
                    <meshStandardMaterial color={pupilColor} />
                </mesh>
            </group>

            {/* Mouth (Simple torus segment or just a box) */}
            <mesh position={[0, 1.1, 0.55]} rotation={[0.2, 0, 0]}>
                <boxGeometry args={[0.3, 0.1, 0.1]} />
                <meshStandardMaterial color="#333" />
            </mesh>

            {/* Hat for Large Pieces? Or maybe just crest for everyone to look like a Gobbler (Rooster) */}
            <group position={[0, 2.5, 0]}>
                <mesh castShadow>
                    <boxGeometry args={[0.1, 0.4, 0.4]} />
                    <meshStandardMaterial color="#ef4444" />
                </mesh>
                <mesh castShadow position={[0, -0.1, 0.3]}>
                    <boxGeometry args={[0.1, 0.2, 0.2]} />
                    <meshStandardMaterial color="#ef4444" />
                </mesh>
            </group>

            {/* Selected Indicator Ring (optional) */}
            {isSelected && (
                <mesh position={[0, 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                    <ringGeometry args={[1, 1.2, 32]} />
                    <meshBasicMaterial color="white" opacity={0.5} transparent />
                </mesh>
            )}
        </animated.group>
    );
}
