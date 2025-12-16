export type Player = 'orange' | 'blue';
export type PieceSize = 'small' | 'medium' | 'large';

export interface Piece {
    id: string;
    player: Player;
    size: PieceSize;
}

export type SelectedPiece = {
    piece: Piece;
    from: 'hand' | { row: number; col: number };
};

export type CellState = Piece[]; // Stack of pieces, last one is visible

export type BoardState = CellState[][]; // 3x3 grid

export interface GameState {
    board: BoardState;
    turn: Player;
    winner: Player | 'draw' | null;
    orangeHand: Piece[];
    blueHand: Piece[];
    selectedPiece: SelectedPiece | null;
}
