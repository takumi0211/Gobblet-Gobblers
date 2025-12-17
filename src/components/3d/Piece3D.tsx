
import { useRef, useMemo, useState, useEffect } from 'react';
import { useSpring, animated, config } from '@react-spring/three';
import { useDrag } from '@use-gesture/react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import type { Piece } from '../../types';

interface Piece3DProps {
    piece: Piece;
    position: [number, number, number];
    isSelected?: boolean;
    onClick?: (e: ThreeEvent<MouseEvent>) => void;
    onDrop?: (row: number, col: number) => void;
    isTop?: boolean; // Only top pieces are interactive
}

export function Piece3D({ piece, position, isSelected, onClick, onDrop, isTop = true }: Piece3DProps) {
    const meshRef = useRef<THREE.Group>(null);
    const { camera, raycaster, scene } = useThree();
    const [isDragging, setIsDragging] = useState(false);

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

    // Animation for position
    const [{ pos }, api] = useSpring(() => ({
        pos: [position[0], position[1], position[2]],
        config: config.wobbly,
    }));

    // Effect to update position when prop changes (if not dragging)
    useEffect(() => {
        if (!isDragging) {
            api.start({
                pos: [position[0], position[1] + (isSelected ? 0.5 : 0), position[2]],
                config: config.wobbly
            });
        }
    }, [position, isSelected, isDragging, api]);

    const bind = useDrag(({ active, timeStamp, event }) => {
        if (!isTop) return;
        event.stopPropagation();

        if (active) {
            setIsDragging(true);
            // Dragging visual logic
            // We need to project mouse to ground plane (approximate)
            // Or simpler: follow mouse on a plane parallel to camera or ground
            const intersection = (event as any).point; // createPointerEvents adds point
            if (intersection) {
                api.start({
                    pos: [intersection.x, 1.5, intersection.z], // Lift up while dragging
                    immediate: true,
                });
            }
        } else {
            setIsDragging(false);
            // On Drag End
            // Raycast to find drop target
            // We can't use event.target usually because the mouse is over the piece itself

            // Manual raycast using mouse position
            // @use-gesture provides xy, but let's use recent pointer from state if available or just check what's under

            // Actually, we can use the raycaster from useThree but we need current mouse coords
            // (event as any).event is the source DOM event.
            // Let's use custom logic to find what's under the mouse *excluding* this piece.

            // Or better: temporary hide this piece, raycast, show it back.
            if (meshRef.current) {
                meshRef.current.visible = false;

                // Raycast
                // Need normalized mouse coords.
                const pointer = (event as any).event ? {
                    x: ((event as any).event.clientX / window.innerWidth) * 2 - 1,
                    y: -((event as any).event.clientY / window.innerHeight) * 2 + 1,
                } : { x: 0, y: 0 };

                raycaster.setFromCamera(new THREE.Vector2(pointer.x, pointer.y), camera);
                const intersects = raycaster.intersectObjects(scene.children, true);

                meshRef.current.visible = true;

                // Find a cell
                const cellHit = intersects.find(hit => hit.object.userData?.type === 'cell');

                if (cellHit && onDrop) {
                    const { row, col } = cellHit.object.userData;
                    onDrop(row, col);
                } else if (cellHit && onClick && !onDrop) {
                    // Fallback if no onDrop provided?
                    // Currently no-op or rely on click
                }
            }

            // Snap back
            api.start({
                pos: [position[0], position[1] + (isSelected ? 0.5 : 0), position[2]],
                config: config.wobbly
            });

            // If it was a click (short drag/tap), trigger onClick
            if (active === false && timeStamp < 200) {
                onClick?.(event as any);
            }
        }
    }, { filterTaps: true }); // We handle taps manually if needed or let filterTaps handle it?
    // Actually filterTaps=true interprets taps as clicks and fires onClick?
    // @use-gesture's onClick is separate.
    // Let's stick to standard useDrag and manually detecting click if needed, or rely on bind() spreading including onClick.
    // For 3D, keeping it simple: useDrag only.

    // REFACTOR PLAN inside this edit:
    // 1. Pass `onDrop` prop to Piece3D?
    // 2. Or trigger specific events via a global store or context?
    // 3. Or just hacking it: The App handles `onCellClick`.
    // If I can call `onCellClick(row, col)` from here that would be great.
    // But I don't have it.

    // I will add `onDrop` prop to Piece3D.


    // Eyes look slightly different for players
    const eyeColor = '#ffffff';
    const pupilColor = '#000000';

    return (
        <animated.group
            ref={meshRef}
            {...(bind() as any)}
            position={pos}
            scale={scale}
        // onClick removed here as it is handled by bind or we add it back if useDrag doesn't capture it?
        // usually useDrag captures clicks if threshold not met. 
        // We can implement click logic in onDragEnd check.
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
