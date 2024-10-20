import random
import numpy as np
from host import GO
from dqn_player import DQNPlayer
from random_player import RandomPlayer
from copy import deepcopy
import torch
from alpha_best import AlphaBetaPlayer
from alpha_beta_aggre import AlphaBetaPlayer1

def train_dqn_player(num_episodes):
    N = 5  # Board size
    dqn_player = DQNPlayer(board_size=N, action_size=N*N+1)
    opponent = AlphaBetaPlayer1()

    for episode in range(num_episodes):
        go = GO(N)
        go.init_board(N)
        piece_type = 1  # DQN player starts first
        go.n_move = 0
        go.X_move = True
        state = dqn_player.get_state(go, piece_type)
        done = False
        while not done:
            if piece_type == 1:
                # DQN player's turn
                action = dqn_player.get_input(go, piece_type)
                # Store the current state and action
                current_state = state
                current_action = action
            else:
                # Opponent's turn
                action = opponent.get_input(go, piece_type)
            if action != "PASS":
                if not go.place_chess(action[0], action[1], piece_type):
                    # Invalid move
                    if piece_type == 1:
                        # Invalid move by DQN player, negative reward
                        reward = -1
                        next_state = None
                        done = True
                        dqn_player.remember(current_state, current_action, reward, next_state, done)
                        break
                    else:
                        # Invalid move by opponent, skip
                        pass
                else:
                    go.died_pieces = go.remove_died_pieces(3 - piece_type)
            else:
                go.previous_board = deepcopy(go.board)
            # Check for game end
            if go.game_end(piece_type, action):
                result = go.judge_winner()
                if result == 1:
                    reward = 1  # DQN player wins
                elif result == 2:
                    reward = -1  # DQN player loses
                else:
                    reward = 0  # Tie
                next_state = None
                done = True
                if piece_type == 1:
                    dqn_player.remember(current_state, current_action, reward, next_state, done)
                break
            # Prepare for next turn
            if piece_type == 1:
                next_state = dqn_player.get_state(go, piece_type)
                reward = 0  # Intermediate reward
                done = False
                dqn_player.remember(current_state, current_action, reward, next_state, done)
                # Train the agent
                dqn_player.train()
            # Switch player
            piece_type = 2 if piece_type == 1 else 1
            if piece_type == 1:
                state = next_state  # Update state only for DQN player

        print(f"Episode {episode+1}/{num_episodes} completed with score {reward}")

    # Save the trained model
    torch.save(dqn_player.model.state_dict(), "dqn_model.pth")

if __name__ == "__main__":
    num_episodes = 1000  # Adjust the number of episodes as needed
    train_dqn_player(num_episodes)