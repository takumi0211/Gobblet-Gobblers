import type { BoardState, Piece, Player } from '../types';

export const getSizeValue = (size: Piece['size']): number => {
    switch (size) {
        case 'small': return 1;
        case 'medium': return 2;
        case 'large': return 3;
        default: return 0;
    }
};

export const cloneBoard = (board: BoardState): BoardState => {
    return board.map(row => row.map(cell => [...cell]));
};

export const getTopPiece = (cell: Piece[]): Piece | null => {
    if (cell.length === 0) return null;
    return cell[cell.length - 1];
};

export const isValidMove = (
    toRow: number,
    toCol: number,
    piece: Piece,
    board: BoardState,
    fromRow?: number,
    fromCol?: number
): boolean => {
    // Cannot move to the same spot
    if (fromRow === toRow && fromCol === toCol) return false;

    const targetCell = board[toRow][toCol];
    const topPiece = getTopPiece(targetCell);

    // If cell is empty, it's valid
    if (!topPiece) return true;

    // If cell has a piece, check if the new piece is larger
    return getSizeValue(piece.size) > getSizeValue(topPiece.size);
};

export const checkWin = (board: BoardState): Player | null => {
    const size = 3;

    // Check rows
    for (let i = 0; i < size; i++) {
        const row = board[i].map(getTopPiece);
        if (row[0] && row[1] && row[2] &&
            row[0].player === row[1].player &&
            row[1].player === row[2].player) {
            return row[0].player;
        }
    }

    // Check cols
    for (let i = 0; i < size; i++) {
        const col = [board[0][i], board[1][i], board[2][i]].map(getTopPiece);
        if (col[0] && col[1] && col[2] &&
            col[0].player === col[1].player &&
            col[1].player === col[2].player) {
            return col[0].player;
        }
    }

    // Check diagonals
    const diag1 = [board[0][0], board[1][1], board[2][2]].map(getTopPiece);
    if (diag1[0] && diag1[1] && diag1[2] &&
        diag1[0].player === diag1[1].player &&
        diag1[1].player === diag1[2].player) {
        return diag1[0].player;
    }

    const diag2 = [board[0][2], board[1][1], board[2][0]].map(getTopPiece);
    if (diag2[0] && diag2[1] && diag2[2] &&
        diag2[0].player === diag2[1].player &&
        diag2[1].player === diag2[2].player) {
        return diag2[0].player;
    }

    return null;
};

export const createInitialBoard = (): BoardState => {
    return Array.from({ length: 3 }, () =>
        Array.from({ length: 3 }, () => [])
    );
};

export const createInitialHand = (player: Player): Piece[] => {
    const pieces: Piece[] = [];
    // 2 of each size
    ['small', 'small', 'medium', 'medium', 'large', 'large'].forEach((size, i) => {
        pieces.push({
            id: `${player}-${size}-${i}`,
            player,
            size: size as Piece['size'],
        });
    });
    return pieces;
};
