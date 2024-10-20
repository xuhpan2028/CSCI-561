# train.py
import numpy as np
import torch
import random
from copy import deepcopy
from host import GO
from dqn_player import DQNPlayer
from alpha_beta_aggre import AlphaBetaPlayer1

def train_dqn(num_episodes=1000, board_size=5):
    # Initialize players
    dqn_player = DQNPlayer(board_size=board_size, action_size=board_size * board_size + 1)
    alpha_beta_player = AlphaBetaPlayer1(depth=3)

    for episode in range(num_episodes):
        # Initialize the game
        go = GO(board_size)
        go.init_board(board_size)
        go.n_move = 0
        go.X_move = True  # Black moves first

        # Randomly assign DQNPlayer's piece_type
        if episode % 2 == 0:
            dqn_piece_type = 1  # DQNPlayer plays black
        else:
            dqn_piece_type = 2  # DQNPlayer plays white

        opponent_piece_type = 3 - dqn_piece_type
        game_over = False

        # Initialize variables to keep track of the game
        while not game_over:
            piece_type = 1 if go.X_move else 2

            if piece_type == dqn_piece_type:
                # DQNPlayer's turn
                state = dqn_player.get_state(go, dqn_piece_type)
                action = dqn_player.get_input(go, dqn_piece_type)

                if action != "PASS":
                    i, j = action
                    valid = go.place_chess(i, j, dqn_piece_type)
                    if not valid:
                        # Invalid move, assign large negative reward and end the game
                        reward = -10
                        next_state = dqn_player.get_state(go, dqn_piece_type)
                        done = True
                        dqn_player.remember(state, action, reward, next_state, done)
                        dqn_player.train()
                        break
                    else:
                        # Check for captures
                        dead_stones = go.remove_died_pieces(opponent_piece_type)
                        reward = len(dead_stones) * 1.0  # Positive reward for capturing stones
                else:
                    # Player passes
                    go.previous_board = deepcopy(go.board)
                    reward = 0

                # Check if game ends
                game_over = go.game_end(dqn_piece_type, action)
                done = game_over

                next_state = dqn_player.get_state(go, dqn_piece_type)
                dqn_player.remember(state, action, reward, next_state, done)
                dqn_player.train()

            else:
                # Opponent's turn
                action = alpha_beta_player.get_input(go, opponent_piece_type)
                if action != "PASS":
                    i, j = action
                    valid = go.place_chess(i, j, opponent_piece_type)
                    if not valid:
                        # Opponent made invalid move, DQNPlayer wins
                        reward = 10
                        next_state = dqn_player.get_state(go, dqn_piece_type)
                        done = True
                        dqn_player.remember(state, action, reward, next_state, done)
                        dqn_player.train()
                        break
                    else:
                        # Remove DQNPlayer's dead stones
                        go.remove_died_pieces(dqn_piece_type)
                else:
                    go.previous_board = deepcopy(go.board)

                # Check if game ends
                game_over = go.game_end(opponent_piece_type, action)

            go.n_move += 1
            go.X_move = not go.X_move  # Switch turns

        # After game ends, assign final reward
        if game_over:
            # Game ended naturally
            winner = go.judge_winner()
            if winner == dqn_piece_type:
                reward = 1  # Win
            elif winner == 0:
                reward = 0  # Tie
            else:
                reward = -1  # Lose
            done = True
            next_state = dqn_player.get_state(go, dqn_piece_type)
            dqn_player.remember(state, action, reward, next_state, done)
            dqn_player.train()

        # Decay epsilon
        if dqn_player.epsilon > dqn_player.epsilon_min:
            dqn_player.epsilon *= dqn_player.epsilon_decay

        # Print progress
        print(f"Episode {episode + 1}/{num_episodes}, value: {reward}")

    # Save the trained model
    torch.save(dqn_player.model.state_dict(), 'dqn_model.pth')
    print("Training completed and model saved.")

if __name__ == "__main__":
    train_dqn(num_episodes=1000, board_size=5)