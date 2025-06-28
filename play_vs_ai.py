import torch
import numpy as np
import random
import os

# train_gobblet.pyから必要なクラスをインポート
from train_gobblet import Piece, GobbletGobblersGame, DQN, ActionMapper

def clear_screen():
    """コンソール画面をクリアする"""
    os.system('cls' if os.name == 'nt' else 'clear')

class HumanPlayer:
    """人間のプレイヤーの入力を処理するクラス"""
    def __init__(self, player_symbol):
        self.player_symbol = player_symbol

    def select_action(self, valid_moves, game):
        """ユーザーに入力を促し、有効な手であればその手を返す"""
        while True:
            # 盤面と手持ちを表示
            clear_screen()
            print("====== ゴブレットゴブラーズ (AI対戦) ======\n")
            game.display()
            print(f"\n--- あなたの番です (プレイヤー '{self.player_symbol}') ---")
            print("アクションを入力してください:")
            print("  - 配置: P <サイズ> <行> <列>  (例: P 3 1 1)")
            print("  - 移動: M <元の行> <元の列> <移動先の行> <移動先の列> (例: M 0 1 2 2)")
            
            action_input = input("入力: ").strip().upper().split()

            try:
                move_type = action_input[0]
                params = [int(p) for p in action_input[1:]]

                if move_type == 'P' and len(params) == 3:
                    move = ('P', params[0], params[1], params[2])
                elif move_type == 'M' and len(params) == 4:
                    move = ('M', params[0], params[1], params[2], params[3])
                else:
                    print("\n[エラー] 入力形式が正しくありません。もう一度入力してください。")
                    input("Enterキーを押して続行...")
                    continue
                
                # 入力された手が有効かどうかをチェック
                if move in valid_moves:
                    return move
                else:
                    print("\n[エラー] その手は無効です。ルールを確認してもう一度入力してください。")
                    input("Enterキーを押して続行...")

            except (ValueError, IndexError):
                print("\n[エラー] 入力形式が正しくありません。数値を正しく入力してください。")
                input("Enterキーを押して続行...")


def play_game():
    """AIと人間が対戦するメインループ"""
    
    # --- 1. セットアップ ---
    env = GobbletGobblersGame()
    action_mapper = ActionMapper()

    # AIエージェントの準備
    ai_agent = DQN(n_observations=120, n_actions=len(action_mapper))
    
    # プレイヤーの選択
    while True:
        human_player_symbol = input("あなたはどちらのプレイヤーになりますか？ ('O' or 'B'): ").strip().upper()
        if human_player_symbol in ['O', 'B']:
            break
        print("無効な入力です。'O'または'B'を入力してください。")

    # AIのシンボルと読み込むモデルを決定
    ai_player_symbol = 'B' if human_player_symbol == 'O' else 'O'
    model_path = f"dqn_gobblet_agent_{ai_player_symbol}.pth"

    try:
        ai_agent.load_state_dict(torch.load(model_path))
        print(f"\nAIモデル '{model_path}' を読み込みました。")
    except FileNotFoundError:
        print(f"[エラー] モデルファイル '{model_path}' が見つかりません。")
        print("先に`train_gobblet.py`を実行して、モデルを学習・保存してください。")
        return

    ai_agent.eval() # 評価モードに設定（学習はしない）

    # プレイヤーを割り当て
    players = {
        human_player_symbol: HumanPlayer(human_player_symbol),
        ai_player_symbol: ai_agent
    }
    
    print("\nゲームを開始します！")
    input("Enterキーを押してスタート...")

    # --- 2. ゲームループ ---
    state = env.reset()
    done = False
    
    while not done:
        current_player_symbol = env.current_player
        
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            print("有効な手がなく、ゲームは引き分けです。")
            break

        move = None
        # 現在のプレイヤーに応じて行動を選択
        if current_player_symbol == human_player_symbol:
            # 人間のターン
            move = players[current_player_symbol].select_action(valid_moves, env)
        else:
            # AIのターン
            clear_screen()
            print("====== ゴブレットゴブラーズ (AI対戦) ======\n")
            env.display()
            print(f"\n--- AI (プレイヤー '{ai_player_symbol}') の思考中... ---")
            
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = ai_agent(state_tensor)[0]
                
                # 有効な手の中からQ値が最大の手を選択
                valid_action_indices = [action_mapper.get_action_index(m) for m in valid_moves]
                mask = torch.full(q_values.shape, -float('inf'))
                mask[valid_action_indices] = 0
                q_values += mask
                
                action_idx = q_values.argmax().item()
                move = action_mapper.get_move(action_idx)
                print(f"AIは手を選びました: {move}")
                input("Enterキーを押して続行...")

        # 選択された手でゲームを進める
        state, reward, done = env.step(move)

    # --- 3. ゲーム終了 ---
    clear_screen()
    print("====== ゲーム終了 ======\n")
    env.display()

    if env.winner:
        if env.winner == human_player_symbol:
            print("\nおめでとうございます！あなたの勝利です！")
        else:
            print(f"\n残念！AI (プレイヤー '{env.winner}') の勝利です。")
    else:
        print("\n引き分けです。")


if __name__ == "__main__":
    play_game()