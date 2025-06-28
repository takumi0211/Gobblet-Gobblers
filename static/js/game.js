class GobbletGobblersGame {
    constructor() {
        this.currentPlayer = 'O';
        this.draggedPiece = null;
        this.dragSource = null;
        this.gameState = null;
        // URL パラメータからモードと色を取得
        const params = new URLSearchParams(window.location.search);
        this.currentMode = params.get('mode') || 'human'; // 'human' or 'ai'
        this.humanColor = params.get('human_color') || 'O';
        
        this.initializeEventListeners();
        // ページ読み込みと同時に新しいゲーム開始
        this.startNewGame(this.currentMode, this.humanColor);
    }

    initializeEventListeners() {
        // （ゲーム画面には存在しない場合がある要素なのでチェック）
        const humanBtn = document.getElementById('start-human-btn');
        if (humanBtn) {
            humanBtn.addEventListener('click', () => {
                this.currentMode = 'human';
                this.humanColor = 'O';
                this.startNewGame(this.currentMode, this.humanColor);
            });
        }

        const aiBtn = document.getElementById('start-ai-btn');
        if (aiBtn) {
            aiBtn.addEventListener('click', () => {
                const colorSelect = document.getElementById('player-color-select');
                const humanColor = colorSelect ? colorSelect.value : 'O';
                this.currentMode = 'ai';
                this.humanColor = humanColor;
                this.startNewGame(this.currentMode, this.humanColor);
        });
        }

        // もう一度プレイボタン (常に存在)
        const playAgainBtn = document.getElementById('play-again-btn');
        if (playAgainBtn) {
            playAgainBtn.addEventListener('click', () => {
                this.startNewGame(this.currentMode, this.humanColor);
            this.hideVictoryModal();
        });
        }

        // ドラッグ&ドロップイベント
        this.setupDragAndDrop();
    }

    setupDragAndDrop() {
        // 手持ちのコマのドラッグ開始
        document.addEventListener('dragstart', (e) => {
            if (e.target.classList.contains('piece') && !e.target.classList.contains('stack-piece')) {
                console.log('手持ちのコマをドラッグ開始:', e.target.dataset);
                this.draggedPiece = e.target;
                this.dragSource = 'hand';
                e.target.classList.add('dragging');
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/html', e.target.outerHTML);
            }
        });

        // 盤上のコマのドラッグ開始
        document.addEventListener('dragstart', (e) => {
            if (e.target.classList.contains('stack-piece')) {
                console.log('盤上のコマをドラッグ開始:', e.target.dataset);
                this.draggedPiece = e.target;
                this.dragSource = 'board';
                e.target.classList.add('dragging');
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/html', e.target.outerHTML);
            }
        });

        // ドラッグ終了
        document.addEventListener('dragend', (e) => {
            if (e.target.classList.contains('piece') || e.target.classList.contains('stack-piece')) {
                e.target.classList.remove('dragging');
                this.draggedPiece = null;
                this.dragSource = null;
            }
        });

        // ドラッグオーバー
        document.addEventListener('dragover', (e) => {
            e.preventDefault();
            // board-cell、piece-stack、stack-pieceのいずれかにドラッグオーバーした場合
            if (e.target.classList.contains('board-cell') || 
                e.target.classList.contains('piece-stack') || 
                e.target.classList.contains('stack-piece')) {
                e.target.classList.add('drag-over');
            }
        });

        // ドラッグリーブ
        document.addEventListener('dragleave', (e) => {
            if (e.target.classList.contains('board-cell') || 
                e.target.classList.contains('piece-stack') || 
                e.target.classList.contains('stack-piece')) {
                e.target.classList.remove('drag-over');
            }
        });

        // ドロップ
        document.addEventListener('drop', (e) => {
            e.preventDefault();
            console.log('ドロップイベント:', e.target);
            
            // ドロップ先のセルを取得
            let targetCell = e.target;
            
            // もしpiece-stackやstack-pieceにドロップした場合、親のboard-cellを取得
            if (e.target.classList.contains('piece-stack') || e.target.classList.contains('stack-piece')) {
                targetCell = e.target.closest('.board-cell');
            }
            
            if (targetCell && targetCell.classList.contains('board-cell')) {
                targetCell.classList.remove('drag-over');
                console.log('ドロップ先のセル:', targetCell.dataset);
                this.handleDrop(targetCell);
            }
        });
    }

    async handleDrop(targetCell) {
        if (!this.draggedPiece) {
            console.log('ドラッグされたコマがありません');
            return;
        }

        const targetRow = parseInt(targetCell.dataset.row);
        const targetCol = parseInt(targetCell.dataset.col);

        console.log('ドロップ処理開始:', {
            dragSource: this.dragSource,
            draggedPiece: this.draggedPiece.dataset,
            targetRow: targetRow,
            targetCol: targetCol
        });

        if (this.dragSource === 'hand') {
            // 手持ちのコマを配置
            const size = parseInt(this.draggedPiece.dataset.size);
            console.log('手持ちのコマを配置:', { size, targetRow, targetCol });
            await this.placePiece(size, targetRow, targetCol);
        } else if (this.dragSource === 'board') {
            // 盤上のコマを移動
            const sourceRow = parseInt(this.draggedPiece.dataset.row);
            const sourceCol = parseInt(this.draggedPiece.dataset.col);
            console.log('盤上のコマを移動:', { sourceRow, sourceCol, targetRow, targetCol });
            await this.movePiece(sourceRow, sourceCol, targetRow, targetCol);
        }
    }

    async loadGameState() {
        try {
            const response = await fetch('/api/get_game_state');
            const data = await response.json();
            this.gameState = data;
            this.updateUI();
        } catch (error) {
            console.error('ゲーム状態の読み込みに失敗しました:', error);
            this.showMessage('ゲーム状態の読み込みに失敗しました', 'error');
        }
    }

    async startNewGame(type = 'human', humanColor = 'O') {
        try {
            let url = '/api/new_game';
            let payload = {};
            if (type === 'ai') {
                url = '/api/new_game_vs_ai';
                payload = { human_player: humanColor };
            }

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (data.status === 'success') {
                await this.loadGameState();
                this.showMessage('新しいゲームを開始しました', 'success');
                
                // AIが先手の場合
                if (data.ai_first) {
                    console.log('AI goes first, starting AI turn');
                    setTimeout(() => {
                        this.startAITurn();
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('新しいゲームの開始に失敗しました:', error);
            this.showMessage('新しいゲームの開始に失敗しました', 'error');
        }
    }

    async placePiece(size, row, col) {
        try {
            const response = await fetch('/api/place_piece', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ size, row, col })
            });
            const data = await response.json();
            
            console.log('placePiece response:', data); // デバッグ用
            
            if (data.status === 'success') {
                await this.loadGameState();
                this.showMessage(data.message, 'success');
                
                if (data.winner) {
                    this.handleVictory(data);
                } else if (data.ai_turn) {
                    console.log('AI turn detected, starting thinking animation'); // デバッグ用
                    // AIのターンが来た場合、思考演出を開始
                    this.startAITurn();
                }
            } else {
                this.showMessage(data.message, 'error');
                this.addInvalidDropEffect(row, col);
            }
        } catch (error) {
            console.error('コマの配置に失敗しました:', error);
            this.showMessage('コマの配置に失敗しました', 'error');
        }
    }

    async movePiece(fromRow, fromCol, toRow, toCol) {
        try {
            const response = await fetch('/api/move_piece', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ from_row: fromRow, from_col: fromCol, to_row: toRow, to_col: toCol })
            });
            const data = await response.json();
            
            console.log('movePiece response:', data); // デバッグ用
            
            if (data.status === 'success') {
                await this.loadGameState();
                this.showMessage(data.message, 'success');
                
                if (data.winner) {
                    this.handleVictory(data);
                } else if (data.ai_turn) {
                    console.log('AI turn detected, starting thinking animation'); // デバッグ用
                    // AIのターンが来た場合、思考演出を開始
                    this.startAITurn();
                }
            } else {
                this.showMessage(data.message, 'error');
                this.addInvalidDropEffect(toRow, toCol);
            }
        } catch (error) {
            console.error('コマの移動に失敗しました:', error);
            this.showMessage('コマの移動に失敗しました', 'error');
        }
    }

    updateUI() {
        if (!this.gameState) return;

        this.updateCurrentPlayer();
        this.updateBoard();
        this.updatePlayerPieces();
        this.updatePlayerLabels();
    }

    updateCurrentPlayer() {
        const playerIndicator = document.getElementById('current-player');
        
        let displayText = `PLAYER ${this.gameState.current_player}`;
        
        // AI対戦の場合は YOU / AI で表示
        if (this.currentMode === 'ai') {
            if (this.gameState.current_player === this.humanColor) {
                displayText = 'YOU';
            } else {
                displayText = 'AI';
            }
        }
        
        playerIndicator.textContent = displayText;
        playerIndicator.className = `player-indicator-game player-${this.gameState.current_player.toLowerCase()}`;
    }

    updateBoard() {
        const board = document.getElementById('board');
        board.innerHTML = '';

        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 3; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.draggable = false;

                const pieces = this.gameState.board[row][col];
                if (pieces.length > 0) {
                    const stack = document.createElement('div');
                    stack.className = 'piece-stack';

                    // コマをサイズ順にソート（小さいコマからappendする）
                    const sortedPieces = [...pieces].sort((a, b) => a.size - b.size);
                    
                    sortedPieces.forEach((piece, index) => {
                        const pieceElement = this.createPieceElement(piece, index);
                        pieceElement.classList.add('stack-piece', `size-${piece.size}`);
                        pieceElement.dataset.row = row;
                        pieceElement.dataset.col = col;
                        pieceElement.draggable = true;
                        stack.appendChild(pieceElement);
                    });

                    cell.appendChild(stack);
                }

                board.appendChild(cell);
            }
        }
    }

    updatePlayerPieces() {
        // プレイヤーOの手持ちコマ
        const playerOPieces = document.getElementById('player-o-pieces');
        playerOPieces.innerHTML = '';
        
        this.gameState.off_board_pieces.O.forEach(piece => {
            const pieceElement = this.createPieceElement(piece);
            pieceElement.draggable = true;
            playerOPieces.appendChild(pieceElement);
        });

        // プレイヤーBの手持ちコマ
        const playerBPieces = document.getElementById('player-b-pieces');
        playerBPieces.innerHTML = '';
        
        this.gameState.off_board_pieces.B.forEach(piece => {
            const pieceElement = this.createPieceElement(piece);
            pieceElement.draggable = true;
            playerBPieces.appendChild(pieceElement);
        });
    }

    createPieceElement(piece, stackIndex = null) {
        const element = document.createElement('div');
        element.className = `piece color-${piece.color.toLowerCase()} size-${piece.size}`;
        element.textContent = `${piece.color}${piece.size}`;
        element.dataset.color = piece.color;
        element.dataset.size = piece.size;
        
        if (stackIndex !== null) {
            element.dataset.stackIndex = stackIndex;
        }

        return element;
    }

    showMessage(message, type = 'info') {
        const messageArea = document.getElementById('message-area');
        messageArea.textContent = message;
        messageArea.className = `message-display ${type} show`;
        
        // 3秒後にメッセージをクリア
        setTimeout(() => {
            messageArea.classList.remove('show');
        setTimeout(() => {
            messageArea.textContent = '';
                messageArea.className = 'message-display';
            }, 300);
        }, 3000);
    }

    addInvalidDropEffect(row, col) {
        const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
        if (cell) {
            cell.classList.add('invalid-drop');
            setTimeout(() => {
                cell.classList.remove('invalid-drop');
            }, 500);
        }
    }

    showVictoryModal(winner) {
        const modal = document.getElementById('victory-modal');
        const message = document.getElementById('victory-message');
        
        let displayText = '';
        if (this.currentMode === 'ai') {
            // 人間 vs AI の場合は勝敗を英語表記で
            displayText = (winner === this.humanColor) ? 'YOU WON!' : 'YOU LOSE...';
        } else {
            const winnerName = winner === 'O' ? 'PLAYER O' : 'PLAYER B';
            displayText = `${winnerName} WINS!`;
        }
        
        message.textContent = displayText;
        
        modal.classList.add('show');
    }

    hideVictoryModal() {
        const modal = document.getElementById('victory-modal');
        modal.classList.remove('show');
    }

    showAIThinkingBeforeMove() {
        console.log('showAIThinkingBeforeMove called'); // デバッグ用
        console.log('currentMode:', this.currentMode, 'humanColor:', this.humanColor); // デバッグ用
        
        // 現在の状態を保存（AI動作前の状態）
        const previousState = JSON.parse(JSON.stringify(this.gameState));
        
        // AIプレイヤーのカードに考えているエフェクトを追加
        const aiPlayerArea = this.currentMode === 'ai' ? 
            (this.humanColor === 'O' ? '.player-b-area' : '.player-o-area') : null;
        
        console.log('aiPlayerArea:', aiPlayerArea); // デバッグ用
        
        if (aiPlayerArea) {
            const aiCard = document.querySelector(`${aiPlayerArea} .player-card`);
            console.log('aiCard found:', !!aiCard); // デバッグ用
            
            if (aiCard) {
                aiCard.classList.add('ai-thinking');
                this.showMessage('AI が考えています...', 'info');
                
                // 1.5秒後にゲーム状態を再読み込みしてアニメーション実行
                setTimeout(async () => {
                    aiCard.classList.remove('ai-thinking');
                    console.log('Thinking finished, reloading game state...');
                    
                    // 新しい状態を取得
                    await this.loadGameState();
                    
                    // アニメーション実行
                    this.animateAIMove(previousState, this.gameState);
                }, 1500);
            }
        } else {
            // AI対戦でない場合（人間同士の場合など）
            console.log('Not AI mode, skipping thinking animation');
        }
    }

    showAIThinking(previousState) {
        // この関数は今後使用しないが、既存のコードとの互換性のため残しておく
        console.log('Legacy showAIThinking called, redirecting to new method');
        this.showAIThinkingBeforeMove();
    }

    animateAIMove(previousState, currentState) {
        console.log('animateAIMove called'); // デバッグ用
        
        // AIが指した手を特定
        const aiMove = this.detectAIMove(previousState, currentState);
        console.log('Detected AI move:', aiMove); // デバッグ用
        
        if (!aiMove) {
            console.log('No AI move detected, updating UI directly');
            this.updateUI();
            return;
        }

        if (aiMove.type === 'place') {
            this.animateAIPlace(aiMove);
        } else if (aiMove.type === 'move') {
            this.animateAIBoardMove(aiMove);
        }
    }

    detectAIMove(previousState, currentState) {
        const aiPlayer = this.currentMode === 'ai' ? 
            (this.humanColor === 'O' ? 'B' : 'O') : null;
        
        console.log('Detecting move for AI player:', aiPlayer); // デバッグ用
        
        if (!aiPlayer) return null;

        // 配置の検出
        const prevOffBoard = previousState.off_board_pieces[aiPlayer];
        const currOffBoard = currentState.off_board_pieces[aiPlayer];
        
        console.log('Previous off-board pieces:', prevOffBoard);
        console.log('Current off-board pieces:', currOffBoard);
        
        if (prevOffBoard.length > currOffBoard.length) {
            // コマが減った = 配置
            const placedPiece = prevOffBoard.find(p => 
                !currOffBoard.some(cp => cp.size === p.size && cp.color === p.color)
            );
            
            console.log('Placed piece:', placedPiece);
            
            // 配置先を探す
            for (let row = 0; row < 3; row++) {
                for (let col = 0; col < 3; col++) {
                    const prevCell = previousState.board[row][col];
                    const currCell = currentState.board[row][col];
                    
                    if (currCell.length > prevCell.length) {
                        const newPiece = currCell[currCell.length - 1];
                        if (newPiece.color === aiPlayer && newPiece.size === placedPiece.size) {
                            console.log(`Found placement at (${row}, ${col})`);
                            return {
                                type: 'place',
                                piece: placedPiece,
                                targetRow: row,
                                targetCol: col
                            };
                        }
                    }
                }
            }
        }

        // 移動の検出
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 3; col++) {
                const prevCell = previousState.board[row][col];
                const currCell = currentState.board[row][col];
                
                if (prevCell.length > currCell.length) {
                    // ここからコマが移動した
                    const movedPiece = prevCell[prevCell.length - 1];
                    if (movedPiece.color === aiPlayer) {
                        // 移動先を探す
                        for (let tr = 0; tr < 3; tr++) {
                            for (let tc = 0; tc < 3; tc++) {
                                if (tr === row && tc === col) continue;
                                const targetPrevCell = previousState.board[tr][tc];
                                const targetCurrCell = currentState.board[tr][tc];
                                
                                if (targetCurrCell.length > targetPrevCell.length) {
                                    const newPiece = targetCurrCell[targetCurrCell.length - 1];
                                    if (newPiece.color === aiPlayer && newPiece.size === movedPiece.size) {
                                        console.log(`Found move from (${row}, ${col}) to (${tr}, ${tc})`);
                                        return {
                                            type: 'move',
                                            piece: movedPiece,
                                            fromRow: row,
                                            fromCol: col,
                                            targetRow: tr,
                                            targetCol: tc
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return null;
    }

    animateAIPlace(aiMove) {
        console.log('animateAIPlace called with:', aiMove); // デバッグ用
        
        // 手持ちエリアから該当するコマを見つける
        const aiPlayer = this.currentMode === 'ai' ? 
            (this.humanColor === 'O' ? 'B' : 'O') : null;
        
        const playerPiecesContainer = document.getElementById(`player-${aiPlayer.toLowerCase()}-pieces`);
        console.log('Player pieces container:', playerPiecesContainer); // デバッグ用
        
        if (!playerPiecesContainer) {
            console.log('Player pieces container not found');
            this.updateUI();
            return;
        }
        
        const pieces = playerPiecesContainer.querySelectorAll('.piece');
        console.log('Available pieces:', pieces); // デバッグ用
        
        let sourcePiece = null;
        for (const piece of pieces) {
            console.log('Checking piece:', piece.dataset); // デバッグ用
            if (parseInt(piece.dataset.size) === aiMove.piece.size) {
                sourcePiece = piece;
                break;
            }
        }

        console.log('Source piece found:', !!sourcePiece); // デバッグ用

        if (!sourcePiece) {
            console.log('Source piece not found, updating UI directly');
            this.updateUI();
            return;
        }

        // ターゲットセル
        const targetCell = document.querySelector(`[data-row="${aiMove.targetRow}"][data-col="${aiMove.targetCol}"]`);
        console.log('Target cell found:', !!targetCell); // デバッグ用
        
        if (!targetCell) {
            console.log('Target cell not found, updating UI directly');
            this.updateUI();
            return;
        }

        this.animatePieceMovement(sourcePiece, targetCell, () => {
            // アニメーション完了後にUIを更新
            console.log('Animation completed, updating UI');
            this.updateUI();
        });
    }

    animateAIBoardMove(aiMove) {
        console.log('animateAIBoardMove called with:', aiMove); // デバッグ用
        
        // 移動元のコマを見つける
        const sourceCell = document.querySelector(`[data-row="${aiMove.fromRow}"][data-col="${aiMove.fromCol}"]`);
        const targetCell = document.querySelector(`[data-row="${aiMove.targetRow}"][data-col="${aiMove.targetCol}"]`);
        
        console.log('Source cell found:', !!sourceCell);
        console.log('Target cell found:', !!targetCell);
        
        if (!sourceCell || !targetCell) {
            console.log('Source or target cell not found, updating UI directly');
            this.updateUI();
            return;
        }

        const sourcePiece = sourceCell.querySelector('.stack-piece:last-child');
        console.log('Source piece found:', !!sourcePiece);
        
        if (!sourcePiece) {
            console.log('Source piece not found, updating UI directly');
            this.updateUI();
            return;
        }

        this.animatePieceMovement(sourcePiece, targetCell, () => {
            // アニメーション完了後にUIを更新
            console.log('Animation completed, updating UI');
            this.updateUI();
        });
    }

    animatePieceMovement(sourcePiece, targetCell, onComplete) {
        console.log('animatePieceMovement called'); // デバッグ用
        
        // ソース要素の位置を取得
        const sourceRect = sourcePiece.getBoundingClientRect();
        const targetRect = targetCell.getBoundingClientRect();

        console.log('Source rect:', sourceRect);
        console.log('Target rect:', targetRect);

        // アニメーション用のクローンを作成
        const animatedPiece = sourcePiece.cloneNode(true);
        animatedPiece.classList.add('ai-move-animation');
        
        // 初期位置を設定
        animatedPiece.style.left = sourceRect.left + 'px';
        animatedPiece.style.top = sourceRect.top + 'px';
        animatedPiece.style.width = sourceRect.width + 'px';
        animatedPiece.style.height = sourceRect.height + 'px';

        document.body.appendChild(animatedPiece);
        console.log('Animated piece added to body');

        // ターゲットセルにエフェクトを追加
        targetCell.classList.add('ai-target');

        // アニメーション開始
        setTimeout(() => {
            console.log('Starting animation');
            animatedPiece.classList.add('moving');
            animatedPiece.style.left = (targetRect.left + targetRect.width/2 - sourceRect.width/2) + 'px';
            animatedPiece.style.top = (targetRect.top + targetRect.height/2 - sourceRect.height/2) + 'px';
        }, 50);

        // アニメーション完了後の処理
        setTimeout(() => {
            console.log('Animation finished, cleaning up');
            if (document.body.contains(animatedPiece)) {
                document.body.removeChild(animatedPiece);
            }
            targetCell.classList.remove('ai-target');
            if (onComplete) onComplete();
        }, 850);
    }

    startAITurn() {
        console.log('startAITurn called'); // デバッグ用
        
        // AIプレイヤーのカードを取得
        const aiPlayerArea = this.currentMode === 'ai' ? 
            (this.humanColor === 'O' ? '.player-b-area' : '.player-o-area') : null;
        
        console.log('aiPlayerArea:', aiPlayerArea); // デバッグ用
        
        if (aiPlayerArea) {
            const aiCard = document.querySelector(`${aiPlayerArea} .player-card`);
            console.log('aiCard found:', !!aiCard); // デバッグ用
            
            if (aiCard) {
                // 思考演出開始
                aiCard.classList.add('ai-thinking');
                this.showMessage('AI が考えています...', 'info');
                
                // 1秒後にAIの手を実行
                setTimeout(() => {
                    aiCard.classList.remove('ai-thinking');
                    this.executeAIMove();
                }, 1000);
            }
        }
    }

    async executeAIMove() {
        console.log('executeAIMove called'); // デバッグ用
        
        try {
            // 現在の状態を保存（アニメーション用）
            const previousState = JSON.parse(JSON.stringify(this.gameState));
            
            const response = await fetch('/api/execute_ai_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();
            
            console.log('executeAIMove response:', data); // デバッグ用
            
            if (data.status === 'success') {
                if (data.ai_move_info) {
                    // アニメーション実行（勝利情報も含める）
                    this.animateAIMoveWithInfo(previousState, data.ai_move_info);
                } else {
                    // フォールバック：通常の更新
                    await this.loadGameState();
                    this.showMessage('AIが手を指しました', 'info');
                }
            } else {
                console.error('AI move failed:', data.message);
                await this.loadGameState();
                this.showMessage(data.message, 'error');
            }
        } catch (error) {
            console.error('AIの手の実行に失敗しました:', error);
            await this.loadGameState();
            this.showMessage('AIの手の実行に失敗しました', 'error');
        }
    }

    animateAIMoveWithInfo(previousState, aiMoveInfo) {
        console.log('animateAIMoveWithInfo called with:', aiMoveInfo); // デバッグ用
        
        if (aiMoveInfo.type === 'place') {
            this.animateAIPlaceWithInfo(aiMoveInfo);
        } else if (aiMoveInfo.type === 'move') {
            this.animateAIBoardMoveWithInfo(aiMoveInfo);
        }
    }

    animateAIPlaceWithInfo(aiMoveInfo) {
        console.log('animateAIPlaceWithInfo called with:', aiMoveInfo); // デバッグ用
        
        // 手持ちエリアから該当するコマを見つける
        const aiPlayer = aiMoveInfo.player;
        const playerPiecesContainer = document.getElementById(`player-${aiPlayer.toLowerCase()}-pieces`);
        
        if (!playerPiecesContainer) {
            console.log('Player pieces container not found');
            this.handleAnimationComplete(aiMoveInfo);
            return;
        }
        
        const pieces = playerPiecesContainer.querySelectorAll('.piece');
        let sourcePiece = null;
        
        for (const piece of pieces) {
            if (parseInt(piece.dataset.size) === aiMoveInfo.size) {
                sourcePiece = piece;
                break;
            }
        }

        if (!sourcePiece) {
            console.log('Source piece not found');
            this.handleAnimationComplete(aiMoveInfo);
            return;
        }

        // ターゲットセル
        const targetCell = document.querySelector(`[data-row="${aiMoveInfo.target_row}"][data-col="${aiMoveInfo.target_col}"]`);
        
        if (!targetCell) {
            console.log('Target cell not found');
            this.handleAnimationComplete(aiMoveInfo);
            return;
        }

        this.animatePieceMovement(sourcePiece, targetCell, () => {
            // アニメーション完了後の処理
            console.log('AI place animation completed');
            this.handleAnimationComplete(aiMoveInfo);
        });
    }

    animateAIBoardMoveWithInfo(aiMoveInfo) {
        console.log('animateAIBoardMoveWithInfo called with:', aiMoveInfo); // デバッグ用
        
        // 移動元のコマを見つける
        const sourceCell = document.querySelector(`[data-row="${aiMoveInfo.from_row}"][data-col="${aiMoveInfo.from_col}"]`);
        const targetCell = document.querySelector(`[data-row="${aiMoveInfo.target_row}"][data-col="${aiMoveInfo.target_col}"]`);
        
        if (!sourceCell || !targetCell) {
            console.log('Source or target cell not found');
            this.handleAnimationComplete(aiMoveInfo);
            return;
        }

        const sourcePiece = sourceCell.querySelector('.stack-piece:last-child');
        
        if (!sourcePiece) {
            console.log('Source piece not found');
            this.handleAnimationComplete(aiMoveInfo);
            return;
        }

        this.animatePieceMovement(sourcePiece, targetCell, () => {
            // アニメーション完了後の処理
            console.log('AI board move animation completed');
            this.handleAnimationComplete(aiMoveInfo);
        });
    }

    async handleAnimationComplete(aiMoveInfo) {
        console.log('handleAnimationComplete called with:', aiMoveInfo); // デバッグ用
        
        // UIを更新
        await this.loadGameState();
        
        // 勝利判定
        if (aiMoveInfo.causes_victory) {
            console.log('AI caused victory, showing winning line and victory modal');
            // 勝利ラインを表示してから勝利モーダル
            this.showWinningLine(aiMoveInfo.winning_line, () => {
                this.showVictoryModal(aiMoveInfo.winner);
            });
        } else {
            this.showMessage('AIが手を指しました', 'info');
        }
    }

    // 勝利ライン演出を表示
    showWinningLine(winningLine, onComplete) {
        if (!winningLine || winningLine.length !== 3) {
            console.log('Invalid winning line:', winningLine);
            if (onComplete) onComplete();
            return;
        }

        console.log('Highlighting winning cells:', winningLine);

        // 勝利したセルを強調
        winningLine.forEach(([row, col]) => {
            const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
            if (cell) {
                cell.classList.add('winning-cell');
            }
        });

        // 2秒後にコールバック実行
        setTimeout(() => {
            if (onComplete) onComplete();
        }, 2000);
    }

    // 勝利ラインを描画
    drawWinningLine(winningLine) {
        const boardContainer = document.querySelector('.board-container');
        if (!boardContainer) return;

        const [[startRow, startCol], [midRow, midCol], [endRow, endCol]] = winningLine;
        
        // ラインの種類を判定
        let lineType = '';
        let lineElement = null;

        if (startRow === endRow) {
            // 横のライン
            lineType = 'horizontal';
            lineElement = this.createHorizontalLine(startRow, boardContainer);
        } else if (startCol === endCol) {
            // 縦のライン
            lineType = 'vertical';
            lineElement = this.createVerticalLine(startCol, boardContainer);
        } else {
            // 斜めのライン
            lineType = 'diagonal';
            lineElement = this.createDiagonalLine(startRow, startCol, endRow, endCol, boardContainer);
        }

        if (lineElement) {
            lineElement.classList.add('winning-line', lineType);
            boardContainer.appendChild(lineElement);
        }
    }

    createHorizontalLine(row, container) {
        const line = document.createElement('div');
        const firstCell = document.querySelector(`[data-row="${row}"][data-col="0"]`);
        const lastCell = document.querySelector(`[data-row="${row}"][data-col="2"]`);
        
        if (firstCell && lastCell) {
            const firstRect = firstCell.getBoundingClientRect();
            const lastRect = lastCell.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            // 左端は行の最初のセルの左端
            line.style.left = (firstRect.left - containerRect.left) + 'px';
            // 行中央に配置
            line.style.top = (firstRect.top - containerRect.top + firstRect.height / 2 - 4) + 'px';
            // 3セル分の幅
            line.style.width = (lastRect.right - firstRect.left) + 'px';
        }
        
        return line;
    }

    createVerticalLine(col, container) {
        const line = document.createElement('div');
        const firstCell = document.querySelector(`[data-row="0"][data-col="${col}"]`);
        const lastCell = document.querySelector(`[data-row="2"][data-col="${col}"]`);
        
        if (firstCell && lastCell) {
            const firstRect = firstCell.getBoundingClientRect();
            const lastRect = lastCell.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            // 列中央に配置
            line.style.left = (firstRect.left - containerRect.left + firstRect.width / 2 - 4) + 'px';
            // 上端は列の最初のセルの上端
            line.style.top = (firstRect.top - containerRect.top) + 'px';
            // 3セル分の高さ
            line.style.height = (lastRect.bottom - firstRect.top) + 'px';
        }
        
        return line;
    }

    createDiagonalLine(startRow, startCol, endRow, endCol, container) {
        const line = document.createElement('div');
        const startCell = document.querySelector(`[data-row="${startRow}"][data-col="${startCol}"]`);
        const endCell = document.querySelector(`[data-row="${endRow}"][data-col="${endCol}"]`);
        
        if (startCell && endCell) {
            const startRect = startCell.getBoundingClientRect();
            const endRect = endCell.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            const startX = startRect.left + startRect.width / 2 - containerRect.left;
            const startY = startRect.top + startRect.height / 2 - containerRect.top;
            const endX = endRect.left + endRect.width / 2 - containerRect.left;
            const endY = endRect.top + endRect.height / 2 - containerRect.top;
            
            const length = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);
            const angle = Math.atan2(endY - startY, endX - startX) * 180 / Math.PI;
            
            line.style.left = startX + 'px';
            line.style.top = (startY - 4) + 'px';
            line.style.width = length + 'px';
            line.style.transformOrigin = '0 50%';
            line.style.transform = `rotate(${angle}deg)`;
        }
        
        return line;
    }

    // 通常の勝利処理（人間同士の対戦用）
    handleVictory(response) {
        console.log('handleVictory called with:', response);
        
        if (response.winning_line) {
            // 勝利ラインを表示してから勝利モーダル
            this.showWinningLine(response.winning_line, () => {
                this.showVictoryModal(response.winner);
            });
        } else {
            // フォールバック：勝利ラインなしで勝利モーダル
            this.showVictoryModal(response.winner);
        }
    }

    // AI対戦の場合にプレイヤーカードのラベルを YOU / AI に切り替える
    updatePlayerLabels() {
        const playerONameEl = document.querySelector('.player-o-area .player-name');
        const playerBNameEl = document.querySelector('.player-b-area .player-name');

        if (!playerONameEl || !playerBNameEl) return;

        if (this.currentMode === 'ai') {
            if (this.humanColor === 'O') {
                playerONameEl.textContent = 'YOU';
                playerBNameEl.textContent = 'AI';
            } else {
                playerONameEl.textContent = 'AI';
                playerBNameEl.textContent = 'YOU';
            }
        } else {
            // 人間同士対戦ではデフォルト表記
            playerONameEl.textContent = 'PLAYER O';
            playerBNameEl.textContent = 'PLAYER B';
        }
    }
}

// ゲームの初期化
document.addEventListener('DOMContentLoaded', () => {
    new GobbletGobblersGame();
}); 