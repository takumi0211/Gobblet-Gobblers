from flask import Flask, render_template, request, jsonify, session
import json
# --- 共通ロジックをtrain_gobbletからインポートしてAIと整合 ---
from train_gobblet import FastGobbletGame as GobbletGobblersGame, Piece, DQN, ActionMapper  # type: ignore
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)
app.secret_key = 'gobblet_gobblers_secret_key'

# 古いモデルとの互換性のためのレガシーDQNクラス
class LegacyDQN(nn.Module):
    """古いモデルファイルとの互換性のためのDQNクラス"""
    
    def __init__(self, n_observations: int, n_actions: int):
        super(LegacyDQN, self).__init__()
        
        # 保存されたモデルの実際の構造に合わせる
        # backbone.0: Linear(120, 128)
        # backbone.1: ReLU
        # backbone.2: Linear(128, 128)  
        # backbone.3: ReLU
        self.backbone = nn.Sequential(
            nn.Linear(n_observations, 128),  # backbone.0
            nn.ReLU(),                       # backbone.1
            nn.Linear(128, 128),             # backbone.2
            nn.ReLU()                        # backbone.3
        )
        self.value_head = nn.Linear(128, n_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.value_head(features)

# === AI関連ユーティリティ ===
ai_agents_cache = {}
action_mapper_global = ActionMapper()

def get_ai_agent(player_symbol: str):
    """プレイヤーシンボル('O' or 'B')に応じたAIエージェントを取得(キャッシュあり)"""
    if player_symbol not in ai_agents_cache:
        model_path = f"dqn_gobblet_agent_{player_symbol}.pth"
        
        try:
            # 最初に新しい構造で試す
            agent = DQN(n_observations=96, n_actions=len(action_mapper_global))
            agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"新しい構造でモデルをロード: {player_symbol}")
        except RuntimeError as e:
            # 新しい構造で失敗した場合、古い構造で試す
            print(f"新しい構造でのロードに失敗、古い構造で試行: {player_symbol}")
            try:
                agent = LegacyDQN(n_observations=120, n_actions=len(action_mapper_global))
                agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"古い構造でモデルをロード成功: {player_symbol}")
            except RuntimeError as e2:
                print(f"モデルロードに失敗: {e2}")
                # ランダムAIとして動作
                agent = LegacyDQN(n_observations=120, n_actions=len(action_mapper_global))
                print(f"ランダムAIとして初期化: {player_symbol}")
        
        agent.eval()
        ai_agents_cache[player_symbol] = agent
    return ai_agents_cache[player_symbol]

def get_state_for_ai(game):
    """状態ベクトルを取得 (モデルに応じて96次元または120次元)"""
    ai_player = session.get('ai_player')
    agent = ai_agents_cache.get(ai_player)
    
    if agent and isinstance(agent, LegacyDQN):
        # 古いモデルの場合は120次元の状態を生成
        return get_legacy_state(game)
    else:
        # 新しいモデルの場合は96次元
        return game._get_state_fast()

def get_legacy_state(game):
    """120次元の古い状態表現を生成"""
    state = np.zeros(120, dtype=np.float32)
    
    # ボード状態（3*3*6 = 54次元）
    for r in range(3):
        for c in range(3):
            base_idx = (r * 3 + c) * 6
            if game.board_state[r, c, 0] > 0:
                player = game.board_state[r, c, 0]
                size = game.board_state[r, c, 1]
                piece_type = (player - 1) * 3 + (size - 1)
                state[base_idx + piece_type] = 1.0
    
    # 手持ちコマ（2*3*2 = 12次元）- 各プレイヤー、各サイズ2個まで
    base_idx = 54
    for size in range(1, 4):
        count_o = min(2, np.sum(game.hand_pieces_o == size))
        count_b = min(2, np.sum(game.hand_pieces_b == size))
        
        # Oプレイヤーの手持ち
        for i in range(2):
            if i < count_o:
                state[base_idx + (size-1)*2 + i] = 1.0
        
        # Bプレイヤーの手持ち
        for i in range(2):
            if i < count_b:
                state[base_idx + 6 + (size-1)*2 + i] = 1.0
    
    # 現在プレイヤー（2次元）
    base_idx = 66
    if game._current_player_int == 1:  # O
        state[base_idx] = 1.0
    else:  # B
        state[base_idx + 1] = 1.0
    
    # 残りの次元は0のまま（合計120次元）
    
    return state

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
    next_state, reward, done = game.step(move)
    return True

def game_state_to_dict(game):
    """FastGobbletGame状態を辞書形式に変換"""
    # board_stateからboard形式に変換
    board_state = []
    for row in range(3):
        board_row = []
        for col in range(3):
            cell_pieces = []
            if game.board_state[row, col, 0] > 0:  # コマがある場合
                color = 'O' if game.board_state[row, col, 0] == 1 else 'B'
                size = int(game.board_state[row, col, 1])
                cell_pieces.append({'color': color, 'size': size})
            board_row.append(cell_pieces)
        board_state.append(board_row)
    
    # hand_piecesからoff_board_pieces形式に変換
    off_board_pieces = {'O': [], 'B': []}
    for size in game.hand_pieces_o:
        if size > 0:
            off_board_pieces['O'].append({'color': 'O', 'size': int(size)})
    for size in game.hand_pieces_b:
        if size > 0:
            off_board_pieces['B'].append({'color': 'B', 'size': int(size)})
    
    return {
        'board': board_state,
        'off_board_pieces': off_board_pieces,
        'current_player': game.current_player,
        'winner': game.winner
    }

def dict_to_game_state(game_dict):
    """辞書形式からFastGobbletGame状態を復元"""
    game = GobbletGobblersGame()
    
    # board_stateを復元
    game.board_state.fill(0)
    for row_idx, row in enumerate(game_dict['board']):
        for col_idx, cell in enumerate(row):
            if cell:  # セルにコマがある場合
                piece_data = cell[-1]  # 一番上のコマ（FastGobbletGameは1つのコマのみ）
                color_int = 1 if piece_data['color'] == 'O' else 2
                game.board_state[row_idx, col_idx, 0] = color_int
                game.board_state[row_idx, col_idx, 1] = piece_data['size']
    
    # hand_piecesを復元
    game.hand_pieces_o.fill(0)
    game.hand_pieces_b.fill(0)
    
    o_pieces = game_dict['off_board_pieces'].get('O', [])
    b_pieces = game_dict['off_board_pieces'].get('B', [])
    
    for i, piece_data in enumerate(o_pieces):
        if i < 6:  # 最大6個まで
            game.hand_pieces_o[i] = piece_data['size']
    
    for i, piece_data in enumerate(b_pieces):
        if i < 6:  # 最大6個まで
            game.hand_pieces_b[i] = piece_data['size']
    
    game.current_player = game_dict['current_player']
    game._current_player_int = 1 if game.current_player == 'O' else 2
    game.winner = game_dict['winner']
    game._cache_valid = False  # キャッシュを無効化
    
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
    
    # 有効な配置かチェック（get_valid_moves()を使用）
    valid_moves = game.get_valid_moves()
    target_move = ('P', size, row, col)
    
    if target_move in valid_moves:
        print(f"配置成功: {game.current_player}{size}を({row},{col})に配置")
        
        # FastGobbletGameのstep()メソッドを使用
        next_state, reward, done = game.step(target_move)
        
        # 勝利判定
        if game.winner:
            win_result = check_win_with_line(game)
            session['game_state'] = game_state_to_dict(game)
            return jsonify({
                'status': 'success',
                'message': f'プレイヤー {game.winner} の勝利！',
                'winner': game.winner,
                'winning_line': win_result['line'] if win_result else None
            })
        
        # --- AIが次のプレイヤーかどうかをチェック ---
        ai_turn = False
        if session.get('vs_ai') and game.current_player == session.get('ai_player'):
            ai_turn = True
        
        session['game_state'] = game_state_to_dict(game)
        return jsonify({'status': 'success', 'message': 'コマを配置しました', 'ai_turn': ai_turn})
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
    
    # 有効な移動かチェック（get_valid_moves()を使用）
    valid_moves = game.get_valid_moves()
    target_move = ('M', from_row, from_col, to_row, to_col)
    
    if target_move in valid_moves:
        print(f"移動成功: {game.current_player}のコマを({from_row},{from_col})から({to_row},{to_col})に移動")
        
        # FastGobbletGameのstep()メソッドを使用
        next_state, reward, done = game.step(target_move)
        
        # 勝利判定
        if game.winner:
            win_result = check_win_with_line(game)
            session['game_state'] = game_state_to_dict(game)
            return jsonify({
                'status': 'success',
                'message': f'プレイヤー {game.winner} の勝利！',
                'winner': game.winner,
                'winning_line': win_result['line'] if win_result else None
            })
        
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
        colors = []
        for r, c in line_coords:
            if game.board_state[r, c, 0] > 0:  # コマがある場合
                colors.append(game.board_state[r, c, 0])
            else:
                colors.append(0)  # 空のマス
        
        # ライン上の全てのマスにコマがあり、かつ全て同じ色か
        if all(color > 0 for color in colors) and all(color == colors[0] for color in colors):
            winner_color = 'O' if colors[0] == 1 else 'B'
            return {
                'winner': winner_color,
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