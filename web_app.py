from flask import Flask, render_template, request, jsonify, session
import json
# --- 共通ロジックをtrain_gobbletからインポートしてAIと整合 ---
from train_gobblet import GobbletGobblersGame, Piece, DQN, ActionMapper  # type: ignore
import torch
import numpy as np

app = Flask(__name__)
app.secret_key = 'gobblet_gobblers_secret_key'

# === AI関連ユーティリティ ===
ai_agents_cache = {}
action_mapper_global = ActionMapper()

def get_ai_agent(player_symbol: str):
    """プレイヤーシンボル('O' or 'B')に応じたAIエージェントを取得(キャッシュあり)"""
    if player_symbol not in ai_agents_cache:
        model_path = f"dqn_gobblet_agent_{player_symbol}.pth"
        agent = DQN(n_observations=120, n_actions=len(action_mapper_global))
        # CPUでロード
        agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.eval()
        ai_agents_cache[player_symbol] = agent
    return ai_agents_cache[player_symbol]

def get_state_for_ai(game):
    """120次元の状態ベクトルを取得 (train_gobbletと同一仕様)"""
    return game._get_state()

def get_valid_moves_for_ai(game):
    """現在プレイヤーの合法手を列挙 (train_gobbletと同一仕様)"""
    return game.get_valid_moves()

def ai_make_move(game):
    """現在のプレイヤーがAIの場合に1手指す。ゲーム状態が変化したらTrueを返す"""
    if not session.get('vs_ai'):
        return False  # AI対戦でない
    ai_player = session.get('ai_player')
    if game.current_player != ai_player:
        return False  # AIの番でない

    agent = get_ai_agent(ai_player)
    state = get_state_for_ai(game)
    valid_moves = get_valid_moves_for_ai(game)
    if not valid_moves:
        return False  # 打てる手がない
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = agent(state_tensor)[0]
        valid_action_indices = [action_mapper_global.get_action_index(m) for m in valid_moves]
        mask = torch.full(q_values.shape, -float('inf'), device=q_values.device)
        mask[valid_action_indices] = 0
        q_values = q_values + mask
        action_idx = q_values.argmax().item()
        move = action_mapper_global.get_move(action_idx)

    # --- 手を適用 ---
    if move[0] == 'P':
        _, size, r, c = move
        piece_to_place = next(p for p in game.off_board_pieces[ai_player] if p.size == size)
        game.off_board_pieces[ai_player].remove(piece_to_place)
        game.board[r][c].append(piece_to_place)
    elif move[0] == 'M':
        _, r_from, c_from, r_to, c_to = move
        moving_piece = game.board[r_from][c_from].pop()
        game.board[r_to][c_to].append(moving_piece)
    # 勝利判定
    game.check_win()
    if not game.winner:
        game.switch_player()
    return True

def game_state_to_dict(game):
    """ゲーム状態を辞書形式に変換"""
    board_state = []
    for row in game.board:
        board_row = []
        for cell in row:
            if cell:
                pieces = [{'color': p.color, 'size': p.size} for p in cell]
                board_row.append(pieces)
            else:
                board_row.append([])
        board_state.append(board_row)
    
    off_board_pieces = {}
    for player, pieces in game.off_board_pieces.items():
        off_board_pieces[player] = [{'color': p.color, 'size': p.size} for p in pieces]
    
    return {
        'board': board_state,
        'off_board_pieces': off_board_pieces,
        'current_player': game.current_player,
        'winner': game.winner
    }

def dict_to_game_state(game_dict):
    """辞書形式からゲーム状態を復元"""
    game = GobbletGobblersGame()
    
    # 盤面を復元
    for row_idx, row in enumerate(game_dict['board']):
        for col_idx, cell in enumerate(row):
            for piece_data in cell:
                piece = Piece(piece_data['color'], piece_data['size'])
                game.board[row_idx][col_idx].append(piece)
    
    # 手持ちのコマを復元
    for player, pieces in game_dict['off_board_pieces'].items():
        game.off_board_pieces[player] = []
        for piece_data in pieces:
            piece = Piece(piece_data['color'], piece_data['size'])
            game.off_board_pieces[player].append(piece)
    
    game.current_player = game_dict['current_player']
    game.winner = game_dict['winner']
    
    # all_piecesを最新のオブジェクトに更新（盤面上と手持ちのすべてのコマ）
    game.all_pieces = []
    # O → B の順でサイズ小→大、同サイズで2個ずつの順に並べる
    for color in ['O', 'B']:
        for size in [1, 1, 2, 2, 3, 3]:
            # まず手持ちから検索
            target_piece = next((p for p in game.off_board_pieces[color] if p.size == size and p not in game.all_pieces), None)
            if target_piece is None:
                # 盤面から検索
                for r in range(3):
                    for c in range(3):
                        target_piece = next((p for p in game.board[r][c] if p.color == color and p.size == size and p not in game.all_pieces), None)
                        if target_piece:
                            break
                    if target_piece:
                        break
            if target_piece:
                game.all_pieces.append(target_piece)
    
    return game

@app.route('/')
def home():
    """ホーム(モード選択)ページを表示"""
    return render_template('home.html')

@app.route('/game')
def game_view():
    """ゲームボード画面を表示"""
    return render_template('index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """新しいゲームを開始 (人間同士)"""
    # 以前の AI 対戦情報をリセット
    session.pop('vs_ai', None)
    session.pop('ai_player', None)
    session.pop('human_player', None)
    game = GobbletGobblersGame()
    session['game_state'] = game_state_to_dict(game)
    return jsonify({'status': 'success', 'message': '新しいゲームを開始しました'})

@app.route('/api/get_game_state', methods=['GET'])
def get_game_state():
    """現在のゲーム状態を取得"""
    if 'game_state' not in session:
        game = GobbletGobblersGame()
        session['game_state'] = game_state_to_dict(game)
    
    return jsonify(session['game_state'])

@app.route('/api/place_piece', methods=['POST'])
def place_piece():
    """手持ちのコマを配置"""
    if 'game_state' not in session:
        return jsonify({'status': 'error', 'message': 'ゲームが開始されていません'})
    
    data = request.get_json()
    size = data.get('size')
    row = data.get('row')
    col = data.get('col')
    
    # 辞書からゲームオブジェクトを復元
    game = dict_to_game_state(session['game_state'])
    
    # デバッグ情報
    print(f"配置試行: サイズ{size}のコマを({row},{col})に配置")
    print(f"現在のプレイヤー: {game.current_player}")
    print(f"手持ちのコマ: {[f'{p.color}{p.size}' for p in game.off_board_pieces[game.current_player]]}")
    
    if game.is_valid_place(size, row, col):
        # 手持ちからコマを削除
        available_pieces = [p for p in game.off_board_pieces[game.current_player] if p.size == size]
        if available_pieces:
            piece_to_place = available_pieces[0]
            game.off_board_pieces[game.current_player].remove(piece_to_place)
            # 盤に配置
            game.board[row][col].append(piece_to_place)
            
            print(f"配置成功: {piece_to_place.color}{piece_to_place.size}を({row},{col})に配置")
            
            # 勝利判定
            win_result = check_win_with_line(game)
            if win_result:
                game.winner = win_result['winner']
                session['game_state'] = game_state_to_dict(game)
                return jsonify({
                    'status': 'success',
                    'message': f'プレイヤー {game.winner} の勝利！',
                    'winner': game.winner,
                    'winning_line': win_result['line']
                })
            
            game.switch_player()
            
            # --- AIが次のプレイヤーかどうかをチェック ---
            ai_turn = False
            if session.get('vs_ai') and game.current_player == session.get('ai_player'):
                ai_turn = True
            
            session['game_state'] = game_state_to_dict(game)
            return jsonify({'status': 'success', 'message': 'コマを配置しました', 'ai_turn': ai_turn})
        else:
            return jsonify({'status': 'error', 'message': '指定されたサイズのコマが見つかりません'})
    else:
        return jsonify({'status': 'error', 'message': '無効な配置です'})

@app.route('/api/move_piece', methods=['POST'])
def move_piece():
    """盤上のコマを移動"""
    if 'game_state' not in session:
        return jsonify({'status': 'error', 'message': 'ゲームが開始されていません'})
    
    data = request.get_json()
    from_row = data.get('from_row')
    from_col = data.get('from_col')
    to_row = data.get('to_row')
    to_col = data.get('to_col')
    
    # 辞書からゲームオブジェクトを復元
    game = dict_to_game_state(session['game_state'])
    
    # デバッグ情報
    print(f"移動試行: ({from_row},{from_col})から({to_row},{to_col})に移動")
    print(f"現在のプレイヤー: {game.current_player}")
    
    if game.is_valid_move(from_row, from_col, to_row, to_col):
        # 移動元のコマを取得して盤から削除
        moving_piece = game.board[from_row][from_col].pop()
        # 移動先に配置
        game.board[to_row][to_col].append(moving_piece)
        
        print(f"移動成功: {moving_piece.color}{moving_piece.size}を({from_row},{from_col})から({to_row},{to_col})に移動")
        
        # 勝利判定
        win_result = check_win_with_line(game)
        if win_result:
            game.winner = win_result['winner']
            session['game_state'] = game_state_to_dict(game)
            return jsonify({
                'status': 'success',
                'message': f'プレイヤー {game.winner} の勝利！',
                'winner': game.winner,
                'winning_line': win_result['line']
            })
        
        game.switch_player()
        
        # --- AIが次のプレイヤーかどうかをチェック ---
        ai_turn = False
        if session.get('vs_ai') and game.current_player == session.get('ai_player'):
            ai_turn = True
        
        session['game_state'] = game_state_to_dict(game)
        return jsonify({'status': 'success', 'message': 'コマを移動しました', 'ai_turn': ai_turn})
    else:
        return jsonify({'status': 'error', 'message': '無効な移動です'})

@app.route('/api/new_game_vs_ai', methods=['POST'])
def new_game_vs_ai():
    """AI対戦モードで新しいゲームを開始。リクエストJSONに human_player ('O' または 'B') を指定可"""
    data = request.get_json(force=True, silent=True) or {}
    human_player = data.get('human_player', 'O').upper()
    if human_player not in ['O', 'B']:
        human_player = 'O'
    ai_player = 'B' if human_player == 'O' else 'O'

    game = GobbletGobblersGame()
    # セッションにAI対戦情報を保存
    session['vs_ai'] = True
    session['human_player'] = human_player
    session['ai_player'] = ai_player

    # AIが先手かどうかをチェック
    ai_first = game.current_player == ai_player

    session['game_state'] = game_state_to_dict(game)
    return jsonify({'status': 'success', 'message': 'AI対戦を開始しました',
                    'human_player': human_player, 'ai_player': ai_player, 'ai_first': ai_first})

@app.route('/api/execute_ai_move', methods=['POST'])
def execute_ai_move():
    """AIの手を実行"""
    if 'game_state' not in session:
        return jsonify({'status': 'error', 'message': 'ゲームが開始されていません'})
    
    if not session.get('vs_ai'):
        return jsonify({'status': 'error', 'message': 'AI対戦モードではありません'})
    
    # 辞書からゲームオブジェクトを復元
    game = dict_to_game_state(session['game_state'])
    
    # AIプレイヤーのターンかチェック
    ai_player = session.get('ai_player')
    if game.current_player != ai_player:
        return jsonify({'status': 'error', 'message': 'AIのターンではありません'})
    
    # AIの手を取得（実行前の状態）
    ai_move_info = get_ai_move_info(game)
    
    # AIの手を実行
    ai_move_made = ai_make_move(game)
    
    if not ai_move_made:
        return jsonify({'status': 'error', 'message': 'AIが手を指せませんでした'})
    
    session['game_state'] = game_state_to_dict(game)
    
    # 勝利情報をai_move_infoに含める（フロントエンドで演出後に表示）
    if game.winner:
        ai_move_info['causes_victory'] = True
        ai_move_info['winner'] = game.winner
        # 勝利ラインを取得
        win_result = check_win_with_line(game)
        if win_result:
            ai_move_info['winning_line'] = win_result['line']
    
    return jsonify({
        'status': 'success', 
        'message': 'AIが手を指しました',
        'ai_move_info': ai_move_info
    })

def check_win_with_line(game):
    """勝利条件をチェックし、勝利ラインがあれば座標を返す"""
    lines_coordinates = [
        # 横のライン
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        # 縦のライン
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        # 斜めのライン
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)]
    ]
    
    for line_coords in lines_coordinates:
        pieces = [game.get_top_piece(r, c) for r, c in line_coords]
        # ライン上の全てのマスにコマがあり、かつ全て同じ色か
        if all(pieces) and all(p.color == pieces[0].color for p in pieces):
            return {
                'winner': pieces[0].color,
                'line': line_coords
            }
    return None

def get_ai_move_info(game):
    """AIが指そうとしている手の情報を取得"""
    ai_player = session.get('ai_player')
    if not ai_player:
        return None
    
    agent = get_ai_agent(ai_player)
    state = get_state_for_ai(game)
    valid_moves = get_valid_moves_for_ai(game)
    
    if not valid_moves:
        return None
    
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = agent(state_tensor)[0]
        valid_action_indices = [action_mapper_global.get_action_index(m) for m in valid_moves]
        mask = torch.full(q_values.shape, -float('inf'), device=q_values.device)
        mask[valid_action_indices] = 0
        q_values = q_values + mask
        action_idx = q_values.argmax().item()
        move = action_mapper_global.get_move(action_idx)
    
    # 手の情報を整理
    if move[0] == 'P':  # Place
        _, size, r, c = move
        return {
            'type': 'place',
            'size': size,
            'target_row': r,
            'target_col': c,
            'player': ai_player
        }
    elif move[0] == 'M':  # Move
        _, r_from, c_from, r_to, c_to = move
        return {
            'type': 'move',
            'from_row': r_from,
            'from_col': c_from,
            'target_row': r_to,
            'target_col': c_to,
            'player': ai_player
        }
    
    return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 