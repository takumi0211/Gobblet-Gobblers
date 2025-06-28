# 🎮 ゴブレットゴブラーズ AI学習プロジェクト

Deep Q-Network（DQN）を使用してゴブレットゴブラーズをプレイするAIエージェントを学習し、**美しいWebアプリケーション**でAIと対戦できるプロジェクトです。

## ✨ 主な機能

- **🤖 AI学習システム**: Double DQNによるセルフプレイ学習
- **🌐 Webアプリケーション**: モダンなUI/UXでブラウザでAIと対戦
- **🖥️ コンソール対戦**: ターミナルでAIと対戦
- **📊 学習進捗の可視化**: 勝率グラフの自動生成
- **🎨 ドラッグ&ドロップ操作**: 直感的なゲーム操作
- **⚡ リアルタイム対戦**: Ajax通信による滑らかなゲーム体験

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
# 依存関係のインストール
pip install -r requirements.txt
```

### 2. AI学習の実行
```bash
# デフォルト設定で学習開始（約30分〜数時間）
python train_gobblet.py

# 短時間での動作確認
python -c "
import train_gobblet
train_gobblet.preset_quick_training()
train_gobblet.train()
"
```

### 3. AIと対戦

**🌐 Webアプリケーション版（推奨）：**
```bash
python web_app.py
# ブラウザで http://localhost:5001 にアクセス
```

**🖥️ コンソール版：**
```bash
python play_vs_ai.py
```

## 🎯 ゲームルール

### 基本ルール
1. **目的**: 3×3の盤面で縦・横・斜めに3つ揃える
2. **コマ**: 各プレイヤーがサイズ1,1,2,2,3,3の計6個のコマを持つ
3. **アクション**: 
   - **配置**: 手持ちのコマを盤面に配置
   - **移動**: 盤上のコマを他のマスに移動
4. **特別ルール**: 大きいコマは小さいコマを覆うことができる（逆は不可）

### 操作方法

**🌐 Webアプリケーション：**
- ホーム画面で「人間 vs AI」または「人間 vs 人間」を選択
- **ドラッグ&ドロップ**: 手持ちエリアから盤面に直感的に配置
- **移動**: 盤上のコマを別のマスにドラッグして移動
- **視覚的フィードバック**: 有効な配置先がハイライト表示
- **AIの手の可視化**: AIの次の手が演出付きで表示

**🖥️ コンソール：**
- 配置: `P <サイズ> <行> <列>` (例: `P 3 1 1`)
- 移動: `M <元の行> <元の列> <移動先の行> <移動先の列>` (例: `M 0 1 2 2`)

## 📁 プロジェクト構成

```
Gobblet-Gobblers/
├── 🧠 AI学習・ゲームロジック
│   ├── train_gobblet.py          # AI学習メインファイル
│   ├── play_vs_ai.py             # コンソール版AI対戦
│   ├── dqn_gobblet_agent_O.pth   # 学習済みモデル（先攻）
│   ├── dqn_gobblet_agent_B.pth   # 学習済みモデル（後攻）
│   └── win_rate_history.png      # 学習進捗グラフ
│
├── 🌐 Webアプリケーション
│   ├── web_app.py                # Flaskアプリケーション
│   ├── templates/                # HTMLテンプレート
│   │   ├── home.html            # モード選択画面
│   │   └── index.html           # ゲーム画面
│   └── static/                  # 静的ファイル
│       ├── css/
│       │   └── style.css        # メインスタイルシート（1273行）
│       └── js/
│           └── game.js          # ゲームロジック（1022行）
│
└── 📄 ドキュメント・設定
    ├── README.md                # このファイル
    ├── requirements.txt         # 依存関係リスト
    └── .gitignore              # Git除外設定
```

## 🌐 Webアプリケーションの機能

### 🎨 UI/UX特徴
- **レスポンシブデザイン**: デスクトップ・タブレット・モバイル対応
- **モダンな見た目**: グラデーション・シャドウ・アニメーション
- **直感的操作**: ドラッグ&ドロップでコマを配置・移動
- **視覚的フィードバック**: 有効な配置先の強調表示
- **滑らかなアニメーション**: コマの移動・配置時の演出

### 🎮 ゲーム機能
- **複数ゲームモード**:
  - 人間 vs AI（O先攻またはB先攻を選択可能）
  - 人間 vs 人間
- **AIの動作可視化**: AIが指す手の演出表示
- **勝利演出**: 勝利ライン（縦・横・斜め）の強調表示
- **ゲーム状態管理**: セッションベースのゲーム継続

### 🔧 技術実装
- **Ajax通信**: ページリロードなしのリアルタイム対戦
- **RESTful API**: JSON形式でのデータ交換
- **セッション管理**: Flask sessionによるゲーム状態保持
- **エラーハンドリング**: 不正な操作の適切な処理

## 🧠 AI学習システム

### アルゴリズム
- **手法**: Double DQN with Experience Replay
- **状態表現**: 120次元ベクトル（各コマの位置をワンホット表現）
- **行動空間**: 配置27通り + 移動72通り = 計99通り
- **報酬設計**: 勝利 +1.0、敗北 -1.0、その他 0.0

### ハイパーパラメータ設定
```python
# 学習設定
NUM_EPISODES = 100000    # 学習エピソード数
LEARNING_RATE = 5e-4     # 学習率
BATCH_SIZE = 128         # バッチサイズ
GAMMA = 0.99            # 割引率

# 探索設定
EPSILON_START = 0.9      # 初期探索率
EPSILON_END = 0.05       # 最終探索率
EPSILON_DECAY = 10000    # 探索率減衰ステップ
```

### 学習設定のカスタマイズ
```python
import train_gobblet

# クイック学習（短時間での動作確認用）
train_gobblet.preset_quick_training()

# 強いAI学習（時間をかけてしっかり学習）
train_gobblet.preset_strong_ai()

# バランス型学習（推奨設定）
train_gobblet.preset_balanced()

# カスタム設定
train_gobblet.update_hyperparameters(
    NUM_EPISODES=50000,
    LEARNING_RATE=1e-4
)

# 学習開始
train_gobblet.train()
```

## 📊 学習結果の確認

### 学習進捗
- 学習完了後、`win_rate_history.png`に勝率の推移が保存されます
- 理想的な学習では、両プレイヤーの勝率が50%前後で安定します

### ログ出力例
```
Episode 10000/100000 | Wins O: 52, B: 48 | Avg Reward O: 0.520, B: 0.480
Episode 20000/100000 | Wins O: 49, B: 51 | Avg Reward O: 0.490, B: 0.510
```

## 🔧 技術仕様

### 🧠 AI学習
- **フレームワーク**: PyTorch 2.0+
- **アルゴリズム**: Double DQN
- **最適化**: AdamW optimizer
- **メモリ**: Experience Replay Buffer
- **計算**: CPU最適化（GPU使用も可能）

### 🌐 Webアプリケーション
- **バックエンド**: Flask（Python）
- **フロントエンド**: 
  - HTML5 + CSS3（1273行のスタイル）
  - JavaScript（1022行のゲームロジック）
  - Ajax通信によるリアルタイム更新
- **API**: RESTful JSON API
- **セッション管理**: Flask session
- **UI**: レスポンシブデザイン + ドラッグ&ドロップ操作

### 🚀 実装済み最適化
- **コマ位置追跡**: O(1)位置検索
- **有効手キャッシュ**: 重複計算の削減  
- **テンソル再利用**: メモリ効率の向上
- **バッチ処理**: 効率的な学習データ処理
- **フロントエンド最適化**: CSSアニメーション + JavaScript最適化

## 🎯 使用例とトラブルシューティング

### よくある問題

**1. モデルファイルが見つからない**
```bash
# 学習を実行してモデルを生成
python train_gobblet.py
```

**2. Webアプリが起動しない**
```bash
# ポートが使用中の場合は別のポートを指定
python web_app.py
# デフォルトは http://localhost:5001
```

**3. 学習が遅い**
```bash
# クイック学習で動作確認
python -c "
import train_gobblet
train_gobblet.preset_quick_training()
train_gobblet.train()
"
```

**4. メモリ不足**
```python
# バッチサイズを小さくする
train_gobblet.update_hyperparameters(BATCH_SIZE=64)
```

### パフォーマンス情報
- **学習時間**: 
  - クイック設定: 約5-10分
  - 標準設定: 約30分-2時間
  - 強いAI設定: 2-8時間
- **メモリ使用量**: 約1-2GB（CPU使用時）
- **必要ディスク容量**: 約50MB
- **Webアプリ**: 軽量（リアルタイム動作）

## 🎮 推奨プレイ方法

1. **初回**: クイック学習でAIを生成
2. **Webブラウザ**: `python web_app.py` でWebアプリ起動
3. **ゲーム選択**: 「人間 vs AI」モードを選択
4. **楽しむ**: ドラッグ&ドロップで直感的に対戦！

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。 