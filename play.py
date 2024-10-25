from host import GO
from random_player import RandomPlayer
from alpha_best import AlphaBetaPlayer
import concurrent.futures
from tqdm import tqdm  # Import tqdm for progress bar
from alpha_beta_player import AlphaBetaPlayer_
from player_greedy import GreedyPlayer
from player_aggressive import AggressivePlayer

# player1 = GreedyPlayer()
player1 = AlphaBetaPlayer_()
player2 = RandomPlayer()

def play_game(N):
    go = GO(N)
    go.verbose = False  # Set to False for faster execution without printing
    result = go.play(player1, player2, verbose=False)
    return result

def play_game_(N):
    go = GO(N)
    go.verbose = False  # Set to False for faster execution without printing
    result = go.play(player2, player1, verbose=False)
    return result

def main():
    
    N = 5  # Size of the Go board
    num_games = 10


    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(play_game, [N] * num_games), total=num_games))
    print("Player 1 win rate as first: ", results.count(1) / len(results))
    print(results)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(play_game_, [N] * num_games), total=num_games))
    print("Player 1 win rate as second: ", results.count(2) / len(results))
    print(results)

if __name__ == "__main__":
    main()