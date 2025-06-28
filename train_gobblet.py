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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import time

# Numbaのインポート（高速化用）
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
    print("Numba利用可能 - JITコンパイレーションを有効化")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba未インストール - 通常のPython実行")
    # Numbaが無い場合のダミーデコレータ
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# =============================================================================
# ハイパーパラメータ設定（最適化版）
# =============================================================================
@dataclass
class HyperParams:
    """学習のハイパーパラメータをまとめて管理するクラス - 最適化版"""
    
    # --- 学習全体の設定 ---
    NUM_EPISODES: int = 20000        # 学習エピソード数
    LOG_INTERVAL: int = 1000         # ログ出力間隔
    
    # --- DQNエージェントの設定（最適化） ---
    GAMMA: float = 0.99                # 割引率
    EPSILON_START: float = 0.9         # 初期ε値（探索率）
    EPSILON_END: float = 0.05          # 最終ε値
    EPSILON_DECAY: int = 10000         # ε減衰ステップ数
    LEARNING_RATE = 2e-3        # 学習率（高速化のため少し上げる）
    BATCH_SIZE = 128            # バッチサイズ（大きくして効率化）
    TAU = 0.005                 # ターゲットネットワーク更新率（小さくして安定化）
    MEMORY_SIZE = 20000         # リプレイバッファサイズ（大きくして安定化）
    
    # --- ニューラルネットワークの設定（最適化） ---
    HIDDEN_SIZE = 256           # 隠れ層のサイズ（少し大きくして表現力向上）
    STATE_DIM = 96              # 状態ベクトルの次元数（最適化で削減）
    NUM_LAYERS = 3              # ネットワーク層数
    
    # --- パフォーマンス最適化の設定 ---
    GRAD_CLIP_VALUE = 1.0       # 勾配クリッピング値（小さくして安定化）
    UPDATE_FREQUENCY = 4        # ネットワーク更新頻度（効率化）
    PRIORITY_REPLAY = False     # 優先度付き経験再生
    DOUBLE_DQN = True          # Double DQN使用フラグ
    
    # --- 新しい最適化パラメータ ---
    USE_VECTORIZED_OPS = True   # ベクトル化操作の使用
    USE_PARALLEL_ENV = True     # 並列環境の使用
    NUM_PARALLEL_ENVS = 4       # 並列環境数
    COMPILED_GAME_LOGIC = NUMBA_AVAILABLE  # コンパイル済みゲームロジック
    USE_COMPACT_STATE = True    # コンパクト状態表現の使用
    CACHE_SIZE = 1000          # LRUキャッシュサイズ

# グローバルにアクセス可能にする
HP = HyperParams()

# =============================================================================
# 高速化用のNumba関数群
# =============================================================================

@njit(cache=True)
def fast_check_win_numba(board_state: np.ndarray) -> int:
    """高速勝利判定 - Numba最適化版
    
    Args:
        board_state: shape (3, 3, 2) の配列 [row, col, (color, size)]
                    color: 0=empty, 1=O, 2=B
    
    Returns:
        0: 勝利なし, 1: O勝利, 2: B勝利
    """
    # 勝利ライン定義（事前計算）
    lines = np.array([
        # 横のライン
        [[0,0], [0,1], [0,2]],
        [[1,0], [1,1], [1,2]],
        [[2,0], [2,1], [2,2]],
        # 縦のライン
        [[0,0], [1,0], [2,0]],
        [[0,1], [1,1], [2,1]],
        [[0,2], [1,2], [2,2]],
        # 斜めのライン
        [[0,0], [1,1], [2,2]],
        [[0,2], [1,1], [2,0]]
    ])
    
    for line_idx in range(8):
        colors = np.zeros(3, dtype=np.int32)
        for pos_idx in range(3):
            r, c = lines[line_idx, pos_idx]
            colors[pos_idx] = board_state[r, c, 0]
        
        if colors[0] > 0 and colors[0] == colors[1] == colors[2]:
            return colors[0]
    
    return 0

@njit(cache=True)
def fast_get_valid_moves_numba(board_state: np.ndarray, 
                               hand_pieces: np.ndarray,
                               current_player: int) -> np.ndarray:
    """高速有効手生成 - Numba最適化版
    
    Args:
        board_state: shape (3, 3, 2) の配列
        hand_pieces: shape (6,) の配列（プレイヤーの手持ちコマサイズ、0=なし）
        current_player: 1=O, 2=B
    
    Returns:
        valid_moves: shape (N, 5) の配列 [type, size, r, c, extra]
                    type: 0=place, 1=move
    """
    moves = []
    
    # 1. 配置（Place）- 手持ちのコマから
    for piece_idx in range(6):
        if hand_pieces[piece_idx] == 0:
            continue
        size = hand_pieces[piece_idx]
        
        for r in range(3):
            for c in range(3):
                # 空きまたは小さいコマの上に置ける
                if board_state[r, c, 0] == 0 or size > board_state[r, c, 1]:
                    moves.append([0, size, r, c, 0])  # type=0(place)
    
    # 2. 移動（Move）- 盤上のコマから
    for r_from in range(3):
        for c_from in range(3):
            if board_state[r_from, c_from, 0] == current_player:
                moving_size = board_state[r_from, c_from, 1]
                
                for r_to in range(3):
                    for c_to in range(3):
                        if r_from == r_to and c_from == c_to:
                            continue
                        
                        # 空きまたは小さいコマの上に移動できる
                        if (board_state[r_to, c_to, 0] == 0 or 
                            moving_size > board_state[r_to, c_to, 1]):
                            moves.append([1, r_from, c_from, r_to, c_to])  # type=1(move)
    
    return np.array(moves, dtype=np.int32) if moves else np.zeros((0, 5), dtype=np.int32)

@njit(cache=True)
def fast_state_encoding_numba(board_state: np.ndarray, 
                              hand_pieces_o: np.ndarray,
                              hand_pieces_b: np.ndarray,
                              current_player: int) -> np.ndarray:
    """高速状態エンコード - Numba最適化版
    
    Returns:
        state: shape (96,) の状態ベクトル（最適化で次元削減）
        - ボード状態: 3*3*6 = 54次元（各マスにつき6種類のコマの有無）
        - 手持ちコマ: 2*6*3 = 36次元（各プレイヤー、各サイズの個数）
        - 現在プレイヤー: 6次元（ワンホット）
        合計: 96次元
    """
    state = np.zeros(96, dtype=np.float32)
    
    # ボード状態のエンコード（54次元）
    for r in range(3):
        for c in range(3):
            base_idx = (r * 3 + c) * 6
            if board_state[r, c, 0] > 0:  # コマがある場合
                player = board_state[r, c, 0]  # 1=O, 2=B
                size = board_state[r, c, 1]    # 1,2,3
                # プレイヤーとサイズを組み合わせたインデックス
                piece_type = (player - 1) * 3 + (size - 1)  # 0-5の範囲
                state[base_idx + piece_type] = 1.0
    
    # 手持ちコマのエンコード（36次元）
    base_idx = 54
    
    # Oプレイヤーの手持ち（18次元）
    for size in range(1, 4):  # サイズ1,2,3
        count = np.sum(hand_pieces_o == size)
        for i in range(6):  # 最大6個まで
            if i < count:
                state[base_idx + (size-1)*6 + i] = 1.0
    
    base_idx += 18
    
    # Bプレイヤーの手持ち（18次元）
    for size in range(1, 4):  # サイズ1,2,3
        count = np.sum(hand_pieces_b == size)
        for i in range(6):  # 最大6個まで
            if i < count:
                state[base_idx + (size-1)*6 + i] = 1.0
    
    # 現在プレイヤー（6次元、ワンホット）
    base_idx = 90
    if current_player == 1:  # O
        state[base_idx:base_idx+3] = 1.0
    else:  # B
        state[base_idx+3:base_idx+6] = 1.0
    
    return state

# =============================================================================
# 最適化されたゲームロジック
# =============================================================================

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

class FastGobbletGame:
    """最適化されたゴブレットゴブラーズゲーム環境"""
    
    # クラス定数
    BOARD_SIZE = 3
    PIECE_SIZES = [1, 1, 2, 2, 3, 3]
    PLAYERS = ['O', 'B']
    PLAYER_TO_INT = {'O': 1, 'B': 2}
    INT_TO_PLAYER = {1: 'O', 2: 'B'}
    
    def __init__(self):
        # NumPy配列による高速データ構造
        self.board_state = np.zeros((3, 3, 2), dtype=np.int8)  # [row, col, (color, size)]
        self.hand_pieces_o = np.array([1, 1, 2, 2, 3, 3], dtype=np.int8)
        self.hand_pieces_b = np.array([1, 1, 2, 2, 3, 3], dtype=np.int8)
        
        # 状態キャッシュ
        self._state_cache = np.zeros(HP.STATE_DIM, dtype=np.float32)
        self._board_cache = np.zeros((3, 3, 2), dtype=np.int8)
        
        # 勝利ライン（事前計算）
        self._winning_lines = self._generate_winning_lines()
        
        # 有効手キャッシュ
        self._valid_moves_cache = None
        self._cache_valid = False
        
        # 互換性のための文字列プレイヤー表現
        self.current_player = 'O'  # 'O', 'B'
        self.winner = None
        self.move_count = 0
        
        # 内部用の数値表現
        self._current_player_int = 1  # 1=O, 2=B
        self._winner_int = 0
        
    @staticmethod
    def _generate_winning_lines() -> np.ndarray:
        """勝利判定用のライン座標を事前生成 - NumPy最適化版"""
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
        return np.array(lines, dtype=np.int8)

    def reset(self) -> np.ndarray:
        """ゲームを初期状態に戻す - 高速版"""
        self.board_state.fill(0)
        self.hand_pieces_o[:] = [1, 1, 2, 2, 3, 3]
        self.hand_pieces_b[:] = [1, 1, 2, 2, 3, 3]
        
        self.current_player = 'O'
        self._current_player_int = 1
        self.winner = None
        self._winner_int = 0
        self.move_count = 0
        self._cache_valid = False
        
        return self._get_state_fast()

    def _get_state_fast(self) -> np.ndarray:
        """高速状態取得"""
        if HP.COMPILED_GAME_LOGIC and NUMBA_AVAILABLE:
            return fast_state_encoding_numba(
                self.board_state, 
                self.hand_pieces_o, 
                self.hand_pieces_b, 
                self._current_player_int
            )
        else:
            return self._get_state_python()
    
    def _get_state_python(self) -> np.ndarray:
        """Python版状態取得（Numba無効時のフォールバック）"""
        state = np.zeros(HP.STATE_DIM, dtype=np.float32)
        
        # ボード状態のエンコード
        for r in range(3):
            for c in range(3):
                base_idx = (r * 3 + c) * 6
                if self.board_state[r, c, 0] > 0:
                    player = self.board_state[r, c, 0]
                    size = self.board_state[r, c, 1]
                    piece_type = (player - 1) * 3 + (size - 1)
                    state[base_idx + piece_type] = 1.0
        
        # 手持ちコマのエンコード
        base_idx = 54
        for size in range(1, 4):
            count_o = np.sum(self.hand_pieces_o == size)
            count_b = np.sum(self.hand_pieces_b == size)
            for i in range(6):
                if i < count_o:
                    state[base_idx + (size-1)*6 + i] = 1.0
                if i < count_b:
                    state[base_idx + 18 + (size-1)*6 + i] = 1.0
        
        # 現在プレイヤー
        base_idx = 90
        if self._current_player_int == 1:
            state[base_idx:base_idx+3] = 1.0
        else:
            state[base_idx+3:base_idx+6] = 1.0
        
        return state

    def get_valid_moves(self) -> List[Tuple]:
        """有効手を高速取得"""
        if self._cache_valid and self._valid_moves_cache is not None:
            return self._valid_moves_cache
        
        hand_pieces = self.hand_pieces_o if self._current_player_int == 1 else self.hand_pieces_b
        
        if HP.COMPILED_GAME_LOGIC and NUMBA_AVAILABLE:
            moves_array = fast_get_valid_moves_numba(
                self.board_state, 
                hand_pieces, 
                self._current_player_int
            )
            # NumPy配列をタプルリストに変換
            moves = []
            for i in range(moves_array.shape[0]):
                move = moves_array[i]
                if move[0] == 0:  # Place
                    moves.append(('P', move[1], move[2], move[3]))
                else:  # Move
                    moves.append(('M', move[1], move[2], move[3], move[4]))
        else:
            moves = self._get_valid_moves_python(hand_pieces)
        
        self._valid_moves_cache = moves
        self._cache_valid = True
        return moves
    
    def _get_valid_moves_python(self, hand_pieces: np.ndarray) -> List[Tuple]:
        """Python版有効手生成"""
        moves = []
        
        # 配置
        available_sizes = sorted(set(size for size in hand_pieces if size > 0))
        for size in available_sizes:
            for r in range(3):
                for c in range(3):
                    if (self.board_state[r, c, 0] == 0 or 
                        size > self.board_state[r, c, 1]):
                        moves.append(('P', size, r, c))
        
        # 移動
        for r_from in range(3):
            for c_from in range(3):
                if self.board_state[r_from, c_from, 0] == self._current_player_int:
                    moving_size = self.board_state[r_from, c_from, 1]
                    for r_to in range(3):
                        for c_to in range(3):
                            if r_from == r_to and c_from == c_to:
                                continue
                            if (self.board_state[r_to, c_to, 0] == 0 or 
                                moving_size > self.board_state[r_to, c_to, 1]):
                                moves.append(('M', r_from, c_from, r_to, c_to))
        
        return moves

    def check_win(self) -> bool:
        """勝利判定 - 高速版"""
        if HP.COMPILED_GAME_LOGIC and NUMBA_AVAILABLE:
            winner_int = fast_check_win_numba(self.board_state)
            if winner_int > 0:
                self._winner_int = winner_int
                self.winner = 'O' if winner_int == 1 else 'B'
                return True
        else:
            # Python版
            for line_coords in self._winning_lines:
                colors = [self.board_state[r, c, 0] for r, c in line_coords]
                if colors[0] > 0 and colors[0] == colors[1] == colors[2]:
                    self._winner_int = colors[0]
                    self.winner = 'O' if colors[0] == 1 else 'B'
                    return True
        
        return False

    def step(self, move: Tuple) -> Tuple[np.ndarray, float, bool]:
        """行動実行 - 高速版"""
        player = self.current_player
        player_int = self._current_player_int
        move_type = move[0]
        
        if move_type == 'P':
            _, size, r, c = move
            # 手持ちコマから除去
            hand_pieces = self.hand_pieces_o if player_int == 1 else self.hand_pieces_b
            for i in range(6):
                if hand_pieces[i] == size:
                    hand_pieces[i] = 0
                    break
            # 盤面に配置
            self.board_state[r, c] = [player_int, size]
            
        elif move_type == 'M':
            _, r_from, c_from, r_to, c_to = move
            # コマを移動
            piece_data = self.board_state[r_from, c_from].copy()
            self.board_state[r_from, c_from] = [0, 0]
            self.board_state[r_to, c_to] = piece_data
        
        self._cache_valid = False
        self.move_count += 1
        
        # 勝利判定
        done = self.check_win()
        reward = 0.0
        if done:
            reward = 1.0 if self.winner == player else -1.0
        
        # プレイヤー切り替え
        self._current_player_int = 2 if self._current_player_int == 1 else 1
        self.current_player = 'O' if self._current_player_int == 1 else 'B'
        
        # 相手に手がない場合
        if not done and len(self.get_valid_moves()) == 0:
            done = True
            reward = 1.0
            self.winner = player
        
        next_state = self._get_state_fast()
        return next_state, reward, done

# GobbletGobblersGameを高速版で置き換える
GobbletGobblersGame = FastGobbletGame

# 下位互換性のためのエイリアス
# 古いバージョンとの互換性を保つ
class LegacyGobbletGobblersGame:
    """旧バージョンとの互換性のためのクラス（使用非推奨）"""
    def __init__(self):
        print("警告: LegacyGobbletGobblersGame は廃止予定です。FastGobbletGame を使用してください。")
        # 高速版に委譲
        self._fast_game = FastGobbletGame()
        # 属性を委譲
        for attr in dir(self._fast_game):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(self._fast_game, attr))

# =============================================================================
# 最適化されたAIコンポーネント
# =============================================================================

Experience = namedtuple('Experience', ('state', 'action_idx', 'reward', 'next_state', 'done'))

class OptimizedReplayBuffer:
    """最適化されたリプレイバッファ - NumPy配列ベース"""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = 0
        self.size = 0
        
        # NumPy配列による高速メモリ管理
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # 高速サンプリング用のインデックス配列
        self._sample_indices = np.arange(capacity, dtype=np.int32)

    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """経験をバッファに追加 - 高速版"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """バッチサンプリング - 高速版"""
        if self.size < batch_size:
            raise ValueError(f"バッファサイズ({self.size})がバッチサイズ({batch_size})より小さいです")
        
        # 高速ランダムサンプリング
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size

class OptimizedDQN(nn.Module):
    """最適化されたDQNネットワーク"""
    
    def __init__(self, n_observations: int, n_actions: int):
        super(OptimizedDQN, self).__init__()
        
        # より効率的なネットワーク構造
        hidden_sizes = [HP.HIDDEN_SIZE] * HP.NUM_LAYERS
        
        layers = []
        input_size = n_observations
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),  # BatchNormの代わりにLayerNorm（推論時安定）
                nn.ReLU(inplace=True),
                nn.Dropout(0.1) if i < len(hidden_sizes) - 1 else nn.Identity()
            ])
            input_size = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_size, n_actions)
        
        # 重み初期化
        self._initialize_weights()
        
        # 推論時最適化
        self.eval_mode_cache = None
    
    def _initialize_weights(self):
        """Xavier初期化による重み設定"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播 - 最適化版"""
        features = self.backbone(x)
        return self.value_head(features)

# 後方互換性のためのエイリアス
ReplayBuffer = OptimizedReplayBuffer
DQN = OptimizedDQN

class OptimizedDQNAgent:
    """最適化されたDQNエージェント"""
    
    def __init__(self, state_dim: int, action_mapper, player_symbol: str):
        self.device = torch.device("cpu")
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

        # 最適化されたネットワーク
        self.policy_net = DQN(state_dim, self.action_dim)
        self.target_net = DQN(state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # より効率的なオプティマイザ
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5,
            amsgrad=True
        )
        
        # 学習率スケジューラ
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=HP.NUM_EPISODES // 4,
            eta_min=self.learning_rate * 0.1
        )
        
        # 最適化されたリプレイバッファ
        self.memory = ReplayBuffer(HP.MEMORY_SIZE, state_dim)
        self.steps_done = 0
        
        # 計算効率化のためのキャッシュ（事前確保）
        self._mask_cache = torch.full((self.action_dim,), -float('inf'), dtype=torch.float32)
        self._state_tensor_cache = torch.zeros(1, state_dim, dtype=torch.float32)
        self._batch_state_cache = torch.zeros(self.batch_size, state_dim, dtype=torch.float32)
        self._batch_action_cache = torch.zeros(self.batch_size, 1, dtype=torch.int64)
        self._batch_reward_cache = torch.zeros(self.batch_size, 1, dtype=torch.float32)
        self._batch_next_state_cache = torch.zeros(self.batch_size, state_dim, dtype=torch.float32)
        self._batch_done_cache = torch.zeros(self.batch_size, dtype=torch.bool)

        # 更新頻度制御
        self.update_counter = 0
        self.training_step = 0
        
        # パフォーマンス統計
        self.inference_times = deque(maxlen=1000)
        self.training_times = deque(maxlen=100)

    def select_action(self, state: np.ndarray, valid_moves: List) -> int:
        """最適化された行動選択 - ε-greedy戦略"""
        start_time = time.perf_counter()
        
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        # 有効行動インデックスの高速取得
        valid_action_indices = [self.action_mapper.get_action_index(m) for m in valid_moves]

        if sample > eps_threshold:
            with torch.no_grad():
                # テンソル再利用（コピーを避ける）
                self._state_tensor_cache[0] = torch.from_numpy(state)
                
                # 推論実行
                self.policy_net.eval()  # 評価モード
                q_values = self.policy_net(self._state_tensor_cache)[0]
                
                # マスク処理の最適化
                mask = self._mask_cache.clone()
                mask[valid_action_indices] = 0.0
                masked_q_values = q_values + mask
                action_idx = masked_q_values.argmax().item()
        else:
            action_idx = random.choice(valid_action_indices)
        
        # パフォーマンス統計
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        
        return action_idx

    def optimize_model(self):
        """最適化されたモデル学習"""
        # 十分な経験が蓄積されるまで待機
        if len(self.memory) < max(self.batch_size, 1000):
            return
        
        start_time = time.perf_counter()
        
        try:
            # 高速バッチサンプリング
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # 事前確保したテンソルを再利用
            self._batch_state_cache[:] = torch.from_numpy(states)
            self._batch_action_cache[:, 0] = torch.from_numpy(actions)
            self._batch_reward_cache[:, 0] = torch.from_numpy(rewards)
            self._batch_next_state_cache[:] = torch.from_numpy(next_states)
            self._batch_done_cache[:] = torch.from_numpy(dones)
            
            # 訓練モードに設定
            self.policy_net.train()
            
            # 現在の状態の価値を計算
            state_action_values = self.policy_net(self._batch_state_cache).gather(1, self._batch_action_cache)
            
            # 次状態の価値を計算
            next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)
            
            # 終了していない状態のみ処理
            non_final_mask = ~self._batch_done_cache
            if non_final_mask.any():
                with torch.no_grad():
                    self.target_net.eval()
                    if HP.DOUBLE_DQN:
                        # Double DQN: Policy networkで行動選択、Target networkで価値評価
                        next_actions = self.policy_net(self._batch_next_state_cache[non_final_mask]).max(1)[1].unsqueeze(1)
                        next_state_values[non_final_mask] = self.target_net(self._batch_next_state_cache[non_final_mask]).gather(1, next_actions).squeeze(1)
                    else:
                        next_state_values[non_final_mask] = self.target_net(self._batch_next_state_cache[non_final_mask]).max(1)[0]
            
            # ベルマン方程式による期待価値
            expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + self._batch_reward_cache
            
            # 損失計算
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            
            # 勾配更新
            self.optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), HP.GRAD_CLIP_VALUE)
            
            self.optimizer.step()
            
            # 学習率スケジューラの更新
            if self.training_step % 100 == 0:
                self.scheduler.step()
            
            self.training_step += 1
            
            # パフォーマンス統計
            training_time = time.perf_counter() - start_time
            self.training_times.append(training_time)
            
        except Exception as e:
            print(f"Warning: 学習中にエラーが発生しました: {e}")
            return

    def update_target_net(self):
        """ターゲットネットワークの更新 - 最適化版"""
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """パフォーマンス統計を取得"""
        return {
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0,
            'avg_training_time': np.mean(self.training_times) if self.training_times else 0.0,
            'memory_usage': len(self.memory),
            'training_steps': self.training_step,
            'epsilon': self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        }

# 後方互換性のためのエイリアス
DQNAgent = OptimizedDQNAgent

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

# =============================================================================
# ヘルパー関数群
# =============================================================================

def print_hyperparameters():
    """現在のハイパーパラメータ設定を表示"""
    print("=" * 60)
    print("最適化版ハイパーパラメータ設定")
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
    print(f"ネットワーク層数:        {HP.NUM_LAYERS}")
    print(f"更新頻度:               {HP.UPDATE_FREQUENCY}")
    print(f"Numba JIT使用:          {HP.COMPILED_GAME_LOGIC}")
    print("=" * 60)

def update_hyperparameters(**kwargs):
    """ハイパーパラメータを動的に更新する関数"""
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
        EPSILON_DECAY=1500,
        BATCH_SIZE=64,
        UPDATE_FREQUENCY=2
    )
    print("クイック学習プリセットを適用しました")

def preset_strong_ai():
    """強いAI学習用のプリセット（時間をかけてしっかり学習）"""
    update_hyperparameters(
        NUM_EPISODES=100000,
        LOG_INTERVAL=1000,
        EPSILON_DECAY=30000,
        LEARNING_RATE=1e-4,
        BATCH_SIZE=256,
        UPDATE_FREQUENCY=1
    )
    print("強いAI学習プリセットを適用しました")

def preset_balanced():
    """バランス型学習用のプリセット（デフォルト設定）"""
    update_hyperparameters(
        NUM_EPISODES=30000,
        LOG_INTERVAL=300,
        EPSILON_DECAY=10000,
        LEARNING_RATE=1e-3,
        BATCH_SIZE=128,
        UPDATE_FREQUENCY=4
    )
    print("バランス型学習プリセットを適用しました")

# =============================================================================
# グラフ描画とパフォーマンス監視
# =============================================================================

def plot_win_rate(win_history, interval):
    """学習の進捗（勝率）をグラフ化して保存する - 最適化版"""
    if not win_history: 
        return
    
    episodes = [(i + 1) * interval for i in range(len(win_history))]
    wins_O = [h['O'] for h in win_history]
    wins_B = [h['B'] for h in win_history]
    total_games = [h['O'] + h['B'] for h in win_history]
    
    win_rate_O = [w / t if t > 0 else 0 for w, t in zip(wins_O, total_games)]
    win_rate_B = [w / t if t > 0 else 0 for w, t in zip(wins_B, total_games)]
    
    # 高性能なプロット設定
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 勝率プロット
    ax1.plot(episodes, win_rate_O, marker='o', linestyle='-', label="Agent 'O' Win Rate", alpha=0.8)
    ax1.plot(episodes, win_rate_B, marker='s', linestyle='--', label="Agent 'B' Win Rate", alpha=0.8)
    ax1.set_title('Win Rate History during Self-Play')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(f'Win Rate over last {interval} episodes')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    
    # 勝利数プロット
    ax2.bar([e - interval/4 for e in episodes], wins_O, width=interval/2, label="Agent 'O' Wins", alpha=0.7)
    ax2.bar([e + interval/4 for e in episodes], wins_B, width=interval/2, label="Agent 'B' Wins", alpha=0.7)
    ax2.set_title('Win Count History')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel(f'Wins in last {interval} episodes')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('win_rate_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n学習の進捗グラフを 'win_rate_history.png' として保存しました。")

def plot_performance_stats(agents: Dict[str, DQNAgent], episode_count: int):
    """パフォーマンス統計をプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (player, agent) in enumerate(agents.items()):
        stats = agent.get_performance_stats()
        
        # 推論時間
        if agent.inference_times:
            axes[0, i].hist(agent.inference_times, bins=50, alpha=0.7)
            axes[0, i].set_title(f'Agent {player} - Inference Time Distribution')
            axes[0, i].set_xlabel('Time (seconds)')
            axes[0, i].set_ylabel('Frequency')
        
        # 訓練時間
        if agent.training_times:
            axes[1, i].plot(agent.training_times, alpha=0.7)
            axes[1, i].set_title(f'Agent {player} - Training Time Trend')
            axes[1, i].set_xlabel('Training Step')
            axes[1, i].set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(f'performance_stats_ep{episode_count}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"パフォーマンス統計を 'performance_stats_ep{episode_count}.png' として保存しました。")

def train():
    """最適化された学習ループ"""
    start_time = time.time()
    num_episodes = HP.NUM_EPISODES
    log_interval = HP.LOG_INTERVAL
    
    # ハイパーパラメータを表示
    print_hyperparameters()
    
    print(f"使用デバイス: CPU")
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"CPU コア数: {mp.cpu_count()}")
    print(f"最適化機能:")
    print(f"  - Numba JIT: {'有効' if NUMBA_AVAILABLE else '無効'}")
    print(f"  - コンパイル済みゲームロジック: {'有効' if HP.COMPILED_GAME_LOGIC else '無効'}")
    print(f"  - 状態ベクトル次元削減: 120 → {HP.STATE_DIM}")
    print(f"  - バッチサイズ: {HP.BATCH_SIZE}")
    print()
    
    # 環境とエージェントの初期化
    env = GobbletGobblersGame()
    action_mapper = ActionMapper()
    
    # プレイヤーIDの正規化
    player_symbols = ['O', 'B'] if hasattr(env, 'PLAYERS') else [1, 2]
    
    agents = {}
    for symbol in player_symbols:
        agents[symbol] = DQNAgent(
            state_dim=HP.STATE_DIM, 
            action_mapper=action_mapper, 
            player_symbol=str(symbol)
        )
    
    # 学習統計
    win_history = []
    wins = {symbol: 0 for symbol in player_symbols}
    episode_rewards = {symbol: [] for symbol in player_symbols}
    episode_lengths = []
    
    # パフォーマンス監視
    episode_times = deque(maxlen=100)
    last_performance_log = 0
    
    print("学習開始...")
    
    for i_episode in tqdm(range(num_episodes), desc="Training episodes", ncols=100):
        episode_start_time = time.time()
        
        # エピソード初期化
        state = env.reset()
        done = False
        step_count = 0
        episode_reward = {symbol: 0 for symbol in player_symbols}

        while not done and step_count < 200:  # 最大ステップ数制限
            current_player = env.current_player
            agent = agents[current_player]
            
            # 有効手の取得
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                done = True
                # 手がないプレイヤーの負け
                other_player = player_symbols[1] if current_player == player_symbols[0] else player_symbols[0]
                wins[other_player] += 1
                continue

            # 現在状態の取得（高速版）
            if hasattr(env, '_get_state_fast'):
                current_state = env._get_state_fast()
            else:
                current_state = env._get_state_for_player(current_player)
            
            # 行動選択
            action_idx = agent.select_action(current_state, valid_moves)
            move = action_mapper.get_move(action_idx)
            
            # 行動実行
            next_state, reward, done = env.step(move)
            
            # 経験の保存
            agent.memory.push(current_state, action_idx, reward, next_state, done)
            
            episode_reward[current_player] += reward
            step_count += 1
            
            # 勝利カウント
            if done and reward == 1.0:
                wins[current_player] += 1

            # モデル更新
            agent.update_counter += 1
            if agent.update_counter % HP.UPDATE_FREQUENCY == 0:
                agent.optimize_model()

        # エピソード統計の記録
        episode_lengths.append(step_count)
        for symbol in player_symbols:
            episode_rewards[symbol].append(episode_reward[symbol])
            # ターゲットネットワーク更新
            agents[symbol].update_target_net()
        
        # パフォーマンス監視
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)

        # ログ出力
        if (i_episode + 1) % log_interval == 0:
            avg_rewards = {symbol: np.mean(episode_rewards[symbol][-log_interval:]) 
                          if episode_rewards[symbol] else 0 for symbol in player_symbols}
            avg_episode_time = np.mean(episode_times) if episode_times else 0
            avg_episode_length = np.mean(episode_lengths[-log_interval:]) if episode_lengths else 0
            
            print(f"\nEpisode {i_episode+1}/{num_episodes}")
            print(f"Wins: {', '.join([f'{s}: {wins[s]}' for s in player_symbols])}")
            print(f"Avg Rewards: {', '.join([f'{s}: {avg_rewards[s]:.3f}' for s in player_symbols])}")
            print(f"Avg Episode Time: {avg_episode_time:.3f}s, Length: {avg_episode_length:.1f} steps")
            
            # エージェントのパフォーマンス統計
            for symbol in player_symbols:
                stats = agents[symbol].get_performance_stats()
                print(f"Agent {symbol} - ε: {stats['epsilon']:.3f}, "
                      f"Mem: {stats['memory_usage']}, "
                      f"Inf: {stats['avg_inference_time']*1000:.2f}ms, "
                      f"Train: {stats['avg_training_time']*1000:.2f}ms")
            
            win_history.append(dict(wins))
            wins = {symbol: 0 for symbol in player_symbols}
        
        # 定期的なパフォーマンス統計の保存
        if (i_episode + 1) % (log_interval * 5) == 0:
            plot_performance_stats(agents, i_episode + 1)

    total_time = time.time() - start_time
    print(f"\n学習完了! 総時間: {total_time:.2f}秒 ({total_time/60:.1f}分)")
    print(f"平均エピソード時間: {total_time/num_episodes:.3f}秒")
    
    # 最終統計の表示
    for symbol in player_symbols:
        final_stats = agents[symbol].get_performance_stats()
        print(f"Agent {symbol} 最終統計:")
        print(f"  - 推論時間: {final_stats['avg_inference_time']*1000:.2f}ms")
        print(f"  - 訓練時間: {final_stats['avg_training_time']*1000:.2f}ms")
        print(f"  - 訓練ステップ: {final_stats['training_steps']:,}")
    
    # グラフ描画
    plot_win_rate(win_history, log_interval)
    plot_performance_stats(agents, num_episodes)

    # モデル保存
    try:
        for symbol in player_symbols:
            filename = f"dqn_gobblet_agent_{symbol}.pth"
            torch.save(agents[symbol].policy_net.state_dict(), filename)
            print(f"モデルを保存しました: {filename}")
    except Exception as e:
        print(f"モデル保存中にエラーが発生しました: {e}")
    
    return agents, win_history

if __name__ == "__main__":
    print("=" * 80)
    print("最適化されたゴブレットゴブラーズ DQN学習システム")
    print("=" * 80)
    
    # パフォーマンステスト
    print("=== パフォーマンステスト ===")
    env = GobbletGobblersGame()
    
    # 状態生成速度テスト
    start_time = time.perf_counter()
    for _ in range(1000):
        state = env.reset()
    state_gen_time = (time.perf_counter() - start_time) * 1000 / 1000
    print(f"状態生成速度: {state_gen_time:.3f}ms/回")
    
    # 初期状態の検証
    initial_state = env.reset()
    print(f"最適化後の状態次元: {initial_state.shape} (従来: 120次元)")
    print(f"状態ベクトルの非ゼロ要素数: {np.count_nonzero(initial_state)}")
    
    # 有効手生成速度テスト
    start_time = time.perf_counter()
    for _ in range(1000):
        valid_moves = env.get_valid_moves()
    move_gen_time = (time.perf_counter() - start_time) * 1000 / 1000
    print(f"有効手生成速度: {move_gen_time:.3f}ms/回")
    print(f"初期の有効手数: {len(valid_moves)}")
    
    # ActionMapperのテスト
    action_mapper = ActionMapper()
    print(f"総行動空間: {len(action_mapper)}")
    
    print("\n=== ハイパーパラメータ設定例 ===")
    print("# 高速テスト用:")
    print("# preset_quick_training()")
    print("#")
    print("# 強力なAI訓練用:")
    print("# preset_strong_ai()")
    print("#")
    print("# バランス型:")
    print("# preset_balanced()")
    print("#")
    print("# カスタム設定:")
    print("# update_hyperparameters(NUM_EPISODES=10000, BATCH_SIZE=256)")
    print()
    
    # メモリ使用量の推定
    estimated_memory = (HP.MEMORY_SIZE * HP.STATE_DIM * 4 * 2) / (1024**2)  # MB
    print(f"推定メモリ使用量: {estimated_memory:.1f} MB")
    
    print("\n=== 最適化機能の確認 ===")
    print(f"Numba JIT: {'✓ 有効' if NUMBA_AVAILABLE else '✗ 無効 (pip install numba で高速化可能)'}")
    print(f"状態ベクトル最適化: ✓ 有効 (120 → {HP.STATE_DIM}次元)")
    print(f"バッチ処理最適化: ✓ 有効 (バッチサイズ: {HP.BATCH_SIZE})")
    print(f"メモリ効率化: ✓ 有効 (NumPy配列ベース)")
    print(f"ネットワーク最適化: ✓ 有効 ({HP.NUM_LAYERS}層, {HP.HIDDEN_SIZE}ユニット)")
    
    # 学習開始のオプション
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            print("\n=== クイック学習モード ===")
            preset_quick_training()
        elif sys.argv[1] == "strong":
            print("\n=== 強力AI訓練モード ===")
            preset_strong_ai()
        elif sys.argv[1] == "balanced":
            print("\n=== バランス型学習モード ===")
            preset_balanced()
    
    print("\n=== 学習開始 ===")
    print("注意: Ctrl+C で中断可能です")
    print()
    
    try:
        agents, history = train()
        print("\n=== 学習完了 ===")
        print("生成されたファイル:")
        print("- dqn_gobblet_agent_O.pth (Oプレイヤーのモデル)")
        print("- dqn_gobblet_agent_B.pth (Bプレイヤーのモデル)")  
        print("- win_rate_history.png (勝率の推移グラフ)")
        print("- performance_stats_*.png (パフォーマンス統計)")
        
    except KeyboardInterrupt:
        print("\n学習が中断されました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()