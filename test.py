from host import GO
from dqn_player import DQNPlayer
from random_player import RandomPlayer
import torch
from host import deepcopy
import concurrent.futures
import tqdm

def play_game():
    N = 5  # Board size
    dqn_player = DQNPlayer(board_size=N, action_size=N*N+1)
    dqn_player.model.load_state_dict(torch.load("dqn_model.pth"))
    dqn_player.model.eval()
    opponent = RandomPlayer()

    go = GO(N)
    result = go.play(dqn_player,opponent,  verbose=False)
    # result = go.play(opponent, dqn_player, verbose=False)
    return result

if __name__ == "__main__":
    N = 5  # Size of the Go board
    num_games = 20

    results = []
    for _ in range(num_games):
        result = play_game()
        results.append(result)


    # win rate
    print("Player 1 win rate: ", results.count(1) / len(results))
    print(results)