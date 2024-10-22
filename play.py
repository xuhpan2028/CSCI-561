from host import GO
from random_player import RandomPlayer
from alpha_best import AlphaBetaPlayer
import concurrent.futures
from tqdm import tqdm  # Import tqdm for progress bar
from alpha_beta_aggre import AlphaBetaPlayer1
from player_mcts import MCTSPlayer
from player_greedy import GreedyPlayer
from player_aggressive import AggressivePlayer

def play_game(N):
    go = GO(N)
    go.verbose = False  # Set to False for faster execution without printing
    player2 = AlphaBetaPlayer1()
    player1 = RandomPlayer()
    result = go.play(player1, player2, verbose=False)
    return result

def main():
    N = 5  # Size of the Go board
    num_games = 20


    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(play_game, [N] * num_games), total=num_games))
    print("Player 1 win rate: ", results.count(1) / len(results))
    print(results)

if __name__ == "__main__":
    main()