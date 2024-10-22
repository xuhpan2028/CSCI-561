import random
import sys
from read import readInput
from write import writeOutput

from host import GO

class GreedyPlayer():
    def __init__(self):
        self.type = 'greedy'

    def get_input(self, go, piece_type):
        '''
        Get one input based on a greedy strategy.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input or "PASS".
        '''        
        possible_placements = []
        max_captured = -1
        best_moves = []

        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    # Copy the board for simulation
                    test_go = go.copy_board()
                    test_go.place_chess(i, j, piece_type)
                    # Remove dead opponent stones
                    dead_stones = test_go.remove_died_pieces(3 - piece_type)
                    num_captured = len(dead_stones)

                    if num_captured > max_captured:
                        max_captured = num_captured
                        best_moves = [(i, j)]
                    elif num_captured == max_captured:
                        best_moves.append((i, j))

        if best_moves:
            return random.choice(best_moves)
        else:
            # If no captures are possible, return "PASS" or a random valid move
            return "PASS"

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = GreedyPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)