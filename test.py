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


def main():
    play_game(5)
if __name__ == "__main__":
    main()