import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm

# =============================================================================
# ハイパーパラメータ設定
# =============================================================================
@dataclass
class HyperParams:
    """学習のハイパーパラメータをまとめて管理するクラス"""
    
    # --- 学習全体の設定 ---
    NUM_EPISODES: int = 30000        # 学習エピソード数
    LOG_INTERVAL: int = 300          # ログ出力間隔
    
    # --- DQNエージェントの設定 ---
    GAMMA: float = 0.99                # 割引率
    EPSILON_START: float = 0.9         # 初期ε値（探索率）
    EPSILON_END: float = 0.05          # 最終ε値
    EPSILON_DECAY: int = 10000         # ε減衰ステップ数
    LEARNING_RATE = 5e-4        # 学習率
    BATCH_SIZE = 128            # バッチサイズ
    TAU = 0.005                 # ターゲットネットワーク更新率
    MEMORY_SIZE = 10000         # リプレイバッファサイズ
    
    # --- ニューラルネットワークの設定 ---
    HIDDEN_SIZE = 128           # 隠れ層のサイズ
    STATE_DIM = 120             # 状態ベクトルの次元数
    
    # --- その他の設定 ---
    GRAD_CLIP_VALUE = 100       # 勾配クリッピング値
    
    # 最適化: 新しいハイパーパラメータ
    UPDATE_FREQUENCY = 4        # ネットワーク更新頻度
    PRIORITY_REPLAY = False     # 優先度付き経験再生（実装時用）
    DOUBLE_DQN = True          # Double DQN使用フラグ

# グローバルにアクセス可能にする
HP = HyperParams()

def print_hyperparameters():
    """現在のハイパーパラメータ設定を表示"""
    print("=" * 60)
    print("現在のハイパーパラメータ設定")
    print("=" * 60)
    print(f"学習エピソード数:        {HP.NUM_EPISODES:,}")
    print(f"ログ出力間隔:           {HP.LOG_INTERVAL:,}")
    print(f"割引率 (γ):             {HP.GAMMA}")
    print(f"初期探索率 (ε_start):    {HP.EPSILON_START}")
    print(f"最終探索率 (ε_end):      {HP.EPSILON_END}")
    print(f"探索率減衰ステップ:      {HP.EPSILON_DECAY:,}")
    print(f"学習率:                 {HP.LEARNING_RATE}")
    print(f"バッチサイズ:           {HP.BATCH_SIZE}")
    print(f"ターゲット更新率 (τ):    {HP.TAU}")
    print(f"リプレイバッファサイズ:  {HP.MEMORY_SIZE:,}")
    print(f"隠れ層サイズ:           {HP.HIDDEN_SIZE}")
    print(f"状態ベクトル次元:        {HP.STATE_DIM}")
    print(f"勾配クリッピング値:      {HP.GRAD_CLIP_VALUE}")
    print("=" * 60)

def update_hyperparameters(**kwargs):
    """ハイパーパラメータを動的に更新する関数
    
    使用例:
    update_hyperparameters(NUM_EPISODES=50000, LEARNING_RATE=1e-3)
    """
    for key, value in kwargs.items():
        if hasattr(HP, key):
            setattr(HP, key, value)
            print(f"更新: {key} = {value}")
        else:
            print(f"警告: {key} は有効なハイパーパラメータではありません")

def preset_quick_training():
    """クイック学習用のプリセット（短時間での動作確認用）"""
    update_hyperparameters(
        NUM_EPISODES=5000,
        LOG_INTERVAL=200,
        EPSILON_DECAY=1500
    )
    print("クイック学習プリセットを適用しました")

def preset_strong_ai():
    """強いAI学習用のプリセット（時間をかけてしっかり学習）"""
    update_hyperparameters(
        NUM_EPISODES=100000,
        LOG_INTERVAL=1000,
        EPSILON_DECAY=30000,
        LEARNING_RATE=1e-4
    )
    print("強いAI学習プリセットを適用しました")

def preset_balanced():
    """バランス型学習用のプリセット（デフォルト設定）"""
    update_hyperparameters(
        NUM_EPISODES=30000,
        LOG_INTERVAL=300,
        EPSILON_DECAY=10000,
        LEARNING_RATE=5e-4
    )
    print("バランス型学習プリセットを適用しました")

# --- 1. ゲームロジック ---
class Piece:
    """コマを表すクラス - メモリ効率化のためslots使用"""
    __slots__ = ('color', 'size', '_hash')
    
    def __init__(self, color: str, size: int):
        self.color = color.upper()
        self.size = size
        self._hash = hash((self.color, self.size))
    
    def __str__(self) -> str: 
        return f"{self.color}{self.size}"
    
    def __repr__(self) -> str: 
        return f"Piece('{self.color}', {self.size})"
    
    def __hash__(self) -> int:
        return self._hash
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Piece):
            return False
        return self.color == other.color and self.size == other.size
    
    def __gt__(self, other):
        if not isinstance(other, Piece): 
            return NotImplemented
        return self.size > other.size

class GobbletGobblersGame:
    """ゴブレットゴブラーズのゲーム環境"""
    
    # クラス定数
    BOARD_SIZE = 3
    PIECE_SIZES = [1, 1, 2, 2, 3, 3]
    PLAYERS = ['O', 'B']
    
    def __init__(self):
        # 勝利判定用のライン定義（事前計算）
        self._winning_lines = self._generate_winning_lines()
        # 状態ベクトル用のキャッシュ
        self._state_cache = np.zeros(HP.STATE_DIM, dtype=np.float32)
        # 🚀 最適化1: コマ位置の直接追跡（O(1)アクセス）
        self._piece_positions = {}  # piece_id -> (row, col) or 'hand'
        # 🚀 最適化2: 有効手のキャッシュ
        self._valid_moves_cache = None
        self._cache_valid = False
        self.reset()

    @staticmethod
    def _generate_winning_lines() -> List[List[Tuple[int, int]]]:
        """勝利判定用のライン座標を事前生成"""
        lines = []
        # 横のライン
        for r in range(3):
            lines.append([(r, c) for c in range(3)])
        # 縦のライン
        for c in range(3):
            lines.append([(r, c) for r in range(3)])
        # 斜めのライン
        lines.append([(i, i) for i in range(3)])
        lines.append([(i, 2 - i) for i in range(3)])
        return lines

    def reset(self):
        """ゲームを初期状態に戻す"""
        self.board = [[[] for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.off_board_pieces = {
            player: [Piece(player, size) for size in self.PIECE_SIZES]
            for player in self.PLAYERS
        }
        # 状態表現用に全コマのリストを作成（合計12個）
        self.all_pieces = self.off_board_pieces['O'] + self.off_board_pieces['B']

        # 🚀 最適化1: コマ位置マップの初期化
        self._piece_positions = {id(piece): 'hand' for piece in self.all_pieces}
        # 🚀 最適化2: キャッシュ無効化
        self._cache_valid = False

        self.current_player = 'O'
        self.winner = None
        # 🔥 重要: 初期状態は最初のプレイヤー視点で返す
        return self._get_state_for_player(self.current_player)

    def get_top_piece(self, row: int, col: int) -> Optional[Piece]:
        """指定されたマスの一番上のコマを返す"""
        return self.board[row][col][-1] if self.board[row][col] else None

    def switch_player(self):
        """プレイヤーを交代する"""
        self.current_player = 'B' if self.current_player == 'O' else 'O'

    def check_win(self) -> bool:
        """勝利条件をチェックする"""
        for line_coords in self._winning_lines:
            pieces = [self.get_top_piece(r, c) for r, c in line_coords]
            if all(pieces) and all(p.color == pieces[0].color for p in pieces):
                self.winner = pieces[0].color
                return True
        return False

    def get_valid_moves(self):
        """現在のプレイヤーが可能な全ての手をリストで返す"""
        # 🚀 最適化: 有効手キャッシュ
        if self._cache_valid and self._valid_moves_cache is not None:
            return self._valid_moves_cache
        
        moves = []
        player = self.current_player

        # 1. 配置 (Place) - 手持ちのコマから重複を除去して効率化
        available_sizes = sorted(set(p.size for p in self.off_board_pieces[player]))
        for size in available_sizes:
            for r in range(self.BOARD_SIZE):
                for c in range(self.BOARD_SIZE):
                    top_piece = self.get_top_piece(r, c)
                    if top_piece is None or size > top_piece.size:
                        moves.append(('P', size, r, c))
        
        # 2. 移動 (Move)
        for r_from in range(self.BOARD_SIZE):
            for c_from in range(self.BOARD_SIZE):
                moving_piece = self.get_top_piece(r_from, c_from)
                if moving_piece and moving_piece.color == player:
                    for r_to in range(self.BOARD_SIZE):
                        for c_to in range(self.BOARD_SIZE):
                            if r_from == r_to and c_from == c_to: continue
                            target_piece = self.get_top_piece(r_to, c_to)
                            if target_piece is None or moving_piece > target_piece:
                                moves.append(('M', r_from, c_from, r_to, c_to))
        
        # 🚀 最適化: キャッシュに保存
        self._valid_moves_cache = moves
        self._cache_valid = True
        return moves
    
    def _get_state(self) -> np.ndarray:
        """ニューラルネットワーク用の状態ベクトルを生成"""
        # 🚀 最適化: O(1)位置アクセスによる高速状態生成
        self._state_cache.fill(0.0)
        
        for idx, piece in enumerate(self.all_pieces):
            position = self._piece_positions[id(piece)]
            if position == 'hand':
                location_idx = 9
            else:
                r, c = position
                location_idx = r * 3 + c
            
            self._state_cache[idx * 10 + location_idx] = 1.0
        
        return self._state_cache.copy()

    def _get_state_for_player(self, player: str) -> np.ndarray:
        """プレイヤー視点の状態ベクトルを生成（自分=1, 相手=-1で区別）"""
        self._state_cache.fill(0.0)
        
        for idx, piece in enumerate(self.all_pieces):
            position = self._piece_positions[id(piece)]
            if position == 'hand':
                location_idx = 9
            else:
                r, c = position
                location_idx = r * 3 + c
            
            # プレイヤー視点で値を設定（自分=1, 相手=-1）
            value = 1.0 if piece.color == player else -1.0
            self._state_cache[idx * 10 + location_idx] = value
        
        return self._state_cache.copy()

    def step(self, move: Tuple) -> Tuple[np.ndarray, float, bool]:
        """行動を実行し、(次の状態, 報酬, 完了フラグ)を返す"""
        player = self.current_player
        move_type = move[0]
        
        if move_type == 'P':
            _, size, r, c = move
            piece_to_place = next(p for p in self.off_board_pieces[player] if p.size == size)
            self.off_board_pieces[player].remove(piece_to_place)
            self.board[r][c].append(piece_to_place)
            # 🚀 最適化: 位置追跡を更新
            self._piece_positions[id(piece_to_place)] = (r, c)
        elif move_type == 'M':
            _, r_from, c_from, r_to, c_to = move
            moving_piece = self.board[r_from][c_from].pop()
            self.board[r_to][c_to].append(moving_piece)
            # 🚀 最適化: 位置追跡を更新
            self._piece_positions[id(moving_piece)] = (r_to, c_to)
        
        # 🚀 最適化: キャッシュ無効化
        self._cache_valid = False
        
        # 🔥 重要: 手を打った直後に勝利判定（プレイヤー切り替え前）
        done = self.check_win()
        reward = 0.0
        if done:
            if self.winner == player:  # 手を打ったプレイヤーが勝利
                reward = 1.0  # 勝利
            else:
                reward = -1.0  # 敗北（あり得ないケースだが安全のため）
        
        # プレイヤー切り替え
        self.switch_player()
        
        # 相手に有効手がない場合の判定
        if not done and len(self.get_valid_moves()) == 0:
            # 相手に手がない = 手を打ったプレイヤーの勝利
            done = True
            reward = 1.0
            self.winner = player

        # 🔥 重要: 次の状態は次のプレイヤー視点で返す
        next_state = self._get_state_for_player(self.current_player)
        return next_state, reward, done

    def display(self):
        """現在の盤面を表示する（デバッグ用）"""
        print("  | 0 | 1 | 2 |")
        print("--+---+---+---+")
        for i, row in enumerate(self.board):
            print(f"{i} |", end="")
            for c, cell in enumerate(row):
                top_piece = self.get_top_piece(i, c)
                piece_str = str(top_piece) if top_piece else ' '
                print(f" {piece_str:<2} |", end="")
            print("\n--+---+---+---+")
        
        print("\n--- 手持ちのコマ ---")
        for player, pieces in self.off_board_pieces.items():
            pieces_str = ', '.join(sorted([str(p) for p in pieces]))
            print(f"プレイヤー {player}: {pieces_str}")
        print("\n--------------------")

# --- 2. AIコンポーネント ---
Experience = namedtuple('Experience', ('state', 'action_idx', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network with optimized architecture"""
    
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # より効率的なネットワーク構造
        self.backbone = nn.Sequential(
            nn.Linear(n_observations, HP.HIDDEN_SIZE),
            nn.ReLU(inplace=True),
            nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE),
            nn.ReLU(inplace=True)
        )
        self.value_head = nn.Linear(HP.HIDDEN_SIZE, n_actions)
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.value_head(features)

class DQNAgent:
    """最適化されたDQNエージェント"""
    
    def __init__(self, state_dim, action_mapper, player_symbol, device=None):
        self.device = device if device is not None else torch.device("cpu")
        self.state_dim = state_dim
        self.action_mapper = action_mapper
        self.action_dim = len(action_mapper)
        self.player_symbol = player_symbol

        # ハイパーパラメータをローカルコピー（アクセス高速化）
        self.gamma = HP.GAMMA
        self.epsilon_start = HP.EPSILON_START
        self.epsilon_end = HP.EPSILON_END
        self.epsilon_decay = HP.EPSILON_DECAY
        self.learning_rate = HP.LEARNING_RATE
        self.batch_size = HP.BATCH_SIZE
        self.tau = HP.TAU

        self.policy_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayBuffer(HP.MEMORY_SIZE)
        self.steps_done = 0
        
        # 計算効率化のためのキャッシュ
        self._mask_cache = torch.full((self.action_dim,), -float('inf'), device=self.device)
        
        # 🚀 最適化: テンソル事前確保
        self._state_tensor_cache = torch.zeros(1, state_dim, device=self.device, dtype=torch.float32)
        self._valid_indices_cache = torch.zeros(self.action_dim, device=self.device, dtype=torch.long)
        
        # 🚀 最適化: 混合精度学習（GPU使用時）
        self.use_amp = device.type == 'cuda' and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # 🚀 最適化: 更新頻度制御
        self.update_counter = 0

    def select_action(self, state: np.ndarray, valid_moves: List) -> int:
        """行動選択 - ε-greedy戦略"""
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        valid_action_indices = [self.action_mapper.get_action_index(m) for m in valid_moves]

        if sample > eps_threshold:
            with torch.no_grad():
                # 🚀 最適化: テンソル再利用
                self._state_tensor_cache[0] = torch.from_numpy(state)
                
                # 🚀 最適化: 混合精度推論
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        q_values = self.policy_net(self._state_tensor_cache)[0]
                else:
                    q_values = self.policy_net(self._state_tensor_cache)[0]
                # マスクを再利用
                mask = self._mask_cache.clone()
                mask[valid_action_indices] = 0
                q_values += mask
                action_idx = q_values.argmax().item()
        else:
            action_idx = random.choice(valid_action_indices)
        return action_idx

    def optimize_model(self):
        """モデルの最適化 - バッチ処理の改善"""
        if len(self.memory) < self.batch_size:
            return
        
        try:
            experiences = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*experiences))

            # バッチテンソルの効率的な作成
            state_batch = torch.tensor(np.vstack(batch.state), dtype=torch.float32, device=self.device)
            action_batch = torch.tensor(batch.action_idx, dtype=torch.int64, device=self.device).unsqueeze(1)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            non_final_mask = torch.tensor([not done for done in batch.done], dtype=torch.bool, device=self.device)
            next_state_values = torch.zeros(self.batch_size, device=self.device)

            # 終了していない状態のみを処理
            if non_final_mask.any():
                non_final_next_states = torch.tensor(
                    np.vstack([batch.next_state[i] for i in range(len(batch.done)) if not batch.done[i]]), 
                    dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    # 🚀 最適化: Double DQN
                    if HP.DOUBLE_DQN:
                        # Policy networkで行動選択、Target networkで価値評価
                        next_actions = self.policy_net(non_final_next_states).argmax(1)
                        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    else:
                        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            
            expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch
            
            # 🚀 最適化: 混合精度学習
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), HP.GRAD_CLIP_VALUE)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), HP.GRAD_CLIP_VALUE)
                self.optimizer.step()
        except Exception as e:
            print(f"Warning: 学習中にエラーが発生しました: {e}")
            return

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

# --- 3. 行動マッピング ---
class ActionMapper:
    def __init__(self):
        self.move_to_idx = {}
        self.idx_to_move = []
        
        # Place
        for size in [1, 2, 3]:
            for r in range(3):
                for c in range(3):
                    move = ('P', size, r, c)
                    if move not in self.move_to_idx:
                        self.move_to_idx[move] = len(self.idx_to_move)
                        self.idx_to_move.append(move)
        # Move
        for r_from in range(3):
            for c_from in range(3):
                for r_to in range(3):
                    for c_to in range(3):
                        if r_from == r_to and c_from == c_to: continue
                        move = ('M', r_from, c_from, r_to, c_to)
                        if move not in self.move_to_idx:
                            self.move_to_idx[move] = len(self.idx_to_move)
                            self.idx_to_move.append(move)

    def get_action_index(self, move): return self.move_to_idx[move]
    def get_move(self, index): return self.idx_to_move[index]
    def __len__(self): return len(self.idx_to_move)

# --- 4. 学習ループとグラフ描画 ---
def plot_win_rate(win_history, interval):
    """学習の進捗（勝率）をグラフ化して保存する"""
    if not win_history: return
    
    episodes = [(i + 1) * interval for i in range(len(win_history))]
    wins_O = [h['O'] for h in win_history]
    wins_B = [h['B'] for h in win_history]
    total_games = [h['O'] + h['B'] for h in win_history]
    
    win_rate_O = [w / t if t > 0 else 0 for w, t in zip(wins_O, total_games)]
    win_rate_B = [w / t if t > 0 else 0 for w, t in zip(wins_B, total_games)]
    
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, win_rate_O, marker='o', linestyle='-', label="Agent 'O' Win Rate (First Player)")
    plt.plot(episodes, win_rate_B, marker='x', linestyle='--', label="Agent 'B' Win Rate (Second Player)")
    
    plt.title('Win Rate History during Self-Play')
    plt.xlabel('Episode')
    plt.ylabel(f'Win Rate over last {interval} episodes')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.xlim(0, episodes[-1] + interval)
    
    plt.savefig('win_rate_history.png')
    print("\n学習の進捗グラフを 'win_rate_history.png' として保存しました。")

def train():
    """最適化された学習ループ"""
    num_episodes = HP.NUM_EPISODES
    log_interval = HP.LOG_INTERVAL
    
    # ハイパーパラメータを表示
    print_hyperparameters()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print(f"PyTorch バージョン: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA バージョン: {torch.version.cuda}")
    
    env = GobbletGobblersGame()
    action_mapper = ActionMapper()
    
    agents = {
        'O': DQNAgent(state_dim=HP.STATE_DIM, action_mapper=action_mapper, player_symbol='O', device=device),
        'B': DQNAgent(state_dim=HP.STATE_DIM, action_mapper=action_mapper, player_symbol='B', device=device)
    }
    
    win_history = []
    wins = {'O': 0, 'B': 0}
    episode_rewards = {'O': [], 'B': []}

    for i_episode in tqdm(range(num_episodes), desc="Training episodes", ncols=100):
        state = env.reset()
        done = False
        
        episode_reward = {'O': 0, 'B': 0}

        while not done:
            player = env.current_player
            agent = agents[player]
            
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                done = True
                opponent = 'B' if player == 'O' else 'O'
                wins[opponent] += 1
                continue

            # 🔥 重要: 現在のプレイヤー視点の状態を取得
            current_state = env._get_state_for_player(player)
            
            action_idx = agent.select_action(state, valid_moves)
            move = action_mapper.get_move(action_idx)
            
            next_state, reward, done = env.step(move)
            
            # 🔥 重要: 現在のプレイヤーの経験として正しく保存
            agent.memory.push(current_state, action_idx, reward, next_state, done)
            
            # 🔥 重要: 相手が敗北した場合、相手にも負の報酬を与える
            if done and reward == 1.0:
                # 勝利したプレイヤーの相手
                opponent = 'B' if player == 'O' else 'O'
                opponent_agent = agents[opponent]
                # 相手の最後の経験に負の報酬を追加（もし経験があれば）
                if len(opponent_agent.memory) > 0:
                    # 最後の経験を取得して負の報酬で更新
                    last_exp = opponent_agent.memory.memory[-1]
                    # 新しい経験として負の報酬を追加
                    # 相手視点の最終状態を取得
                    final_state_for_opponent = env._get_state_for_player(opponent)
                    opponent_agent.memory.push(last_exp.state, last_exp.action_idx, -1.0, final_state_for_opponent, True)
            
            episode_reward[player] += reward
            
            # 🔥 修正: 勝利カウントの正確な記録
            if done and reward == 1.0:
                wins[player] += 1

            # 🔥 重要: 次のループのために状態を更新（次のプレイヤー視点）
            if not done:
                state = next_state
            else:
                state = current_state  # ゲーム終了時は現在の状態を保持

            # 🚀 最適化: 更新頻度制御
            agent.update_counter += 1
            if agent.update_counter % HP.UPDATE_FREQUENCY == 0:
                agent.optimize_model()

        for player in ['O', 'B']:
            episode_rewards[player].append(episode_reward[player])
        
        for p_symbol in ['O', 'B']:
            agents[p_symbol].update_target_net()

        if (i_episode + 1) % log_interval == 0:
            avg_reward_O = np.mean(episode_rewards['O'][-log_interval:]) if episode_rewards['O'] else 0
            avg_reward_B = np.mean(episode_rewards['B'][-log_interval:]) if episode_rewards['B'] else 0
            print(f"Episode {i_episode+1}/{num_episodes} | Wins O: {wins['O']}, B: {wins['B']} | Avg Reward O: {avg_reward_O:.3f}, B: {avg_reward_B:.3f}")
            win_history.append({'O': wins['O'], 'B': wins['B']})
            wins = {'O': 0, 'B': 0}

    print("\nTraining finished.")
    
    plot_win_rate(win_history, log_interval)

    # モデル保存
    try:
        torch.save(agents['O'].policy_net.state_dict(), "dqn_gobblet_agent_O.pth")
        torch.save(agents['B'].policy_net.state_dict(), "dqn_gobblet_agent_B.pth")
        print("モデルを保存しました: dqn_gobblet_agent_O.pth, dqn_gobblet_agent_B.pth")
    except Exception as e:
        print(f"モデル保存中にエラーが発生しました: {e}")
    
    return agents, win_history

if __name__ == "__main__":
    # テスト用のコード
    print("=== ハイパーパラメータ設定例 ===")
    print("# 学習時間を短くしたい場合:")
    print("# preset_quick_training()  # または update_hyperparameters(NUM_EPISODES=5000)")
    print("# ")
    print("# より強いAIを作りたい場合:")
    print("# preset_strong_ai()  # または update_hyperparameters(NUM_EPISODES=100000)")
    print("# ")
    print("# バランス型学習の場合:")
    print("# preset_balanced()  # デフォルト設定")
    print("# ")
    print("# 現在の設定を変更する場合は、train()を呼ぶ前にupdate_hyperparameters()を使用")
    print()
    
    print("=== 状態表現のテスト ===")
    env = GobbletGobblersGame()
    
    # 初期状態をテスト
    initial_state = env.reset()
    print(f"初期状態の次元: {initial_state.shape}")
    print(f"初期状態の合計: {initial_state.sum()}")  # 12個のコマが手持ちにあるので12になるはず
    
    # 手を打ってみる
    valid_moves = env.get_valid_moves()
    print(f"初期の有効手数: {len(valid_moves)}")
    
    # 1手打つ
    first_move = valid_moves[0]
    print(f"最初の手: {first_move}")
    next_state, reward, done = env.step(first_move)
    print(f"1手後の状態の合計: {next_state.sum()}")  # まだ12になるはず
    print(f"報酬: {reward}, 終了: {done}")
    
    # ActionMapperのテスト
    action_mapper = ActionMapper()
    print(f"総行動数: {len(action_mapper)}")
    
    print("\n=== 学習開始 ===")
    train()