import random
import sys
from read import readInput
from write import writeOutput
from host import GO

# Define transformation functions
def identity(board):
    return board

def rotate90(board):
    N = len(board)
    return [[board[N - 1 - j][i] for j in range(N)] for i in range(N)]

def rotate180(board):
    N = len(board)
    return [[board[N - 1 - i][N - 1 - j] for j in range(N)] for i in range(N)]

def rotate270(board):
    N = len(board)
    return [[board[j][N - 1 - i] for j in range(N)] for i in range(N)]

def reflect_x(board):
    return board[::-1]

def reflect_y(board):
    return [row[::-1] for row in board]

def reflect_main_diag(board):
    N = len(board)
    return [[board[j][i] for j in range(N)] for i in range(N)]

def reflect_anti_diag(board):
    N = len(board)
    return [[board[N - 1 - j][N - 1 - i] for j in range(N)] for i in range(N)]

# Define move transformation functions
def move_identity(i, j, N):
    return i, j

def move_rotate90(i, j, N):
    return j, N - 1 - i

def move_rotate180(i, j, N):
    return N - 1 - i, N - 1 - j

def move_rotate270(i, j, N):
    return N - 1 - j, i

def move_reflect_x(i, j, N):
    return N - 1 - i, j

def move_reflect_y(i, j, N):
    return i, N - 1 - j

def move_reflect_main_diag(i, j, N):
    return j, i

def move_reflect_anti_diag(i, j, N):
    return N - 1 - j, N - 1 - i

# Map transformations to their inverses
transformations = [
    (identity, move_identity, move_identity),
    (rotate90, move_rotate90, move_rotate270),
    (rotate180, move_rotate180, move_rotate180),
    (rotate270, move_rotate270, move_rotate90),
    (reflect_x, move_reflect_x, move_reflect_x),
    (reflect_y, move_reflect_y, move_reflect_y),
    (reflect_main_diag, move_reflect_main_diag, move_reflect_main_diag),
    (reflect_anti_diag, move_reflect_anti_diag, move_reflect_anti_diag)
]

class RandomPlayer():
    def __init__(self):
        self.type = 'random'
        self.transposition_table = {
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]": (2, 2),
            #"[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]": (3, 3),
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0]]": (3, 3),
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]]": (3, 2),
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]]": (3, 2),
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 2, 0], [0, 0, 1, 2, 0], [0, 0, 0, 0, 0]]": (1, 3),
            "[[0, 0, 0, 0, 0], [0, 2, 1, 1, 0], [0, 0, 2, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]": (2, 1),


            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]": (3, 2),
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 2, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]": (1, 1),
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 2, 2, 1, 0], [0, 0, 0, 0, 0]]": (2, 1),
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 2, 2, 1, 0], [0, 0, 0, 0, 0]]": (2, 3),
            "[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 2, 2, 1, 0], [0, 1, 1, 2, 0], [0, 0, 0, 0, 0]]": (4, 3),
            "[[0, 0, 0, 0, 0], [0, 1, 2, 2, 0], [0, 2, 1, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]": (0, 1)
        }

    def get_input(self, go, piece_type):
        N = go.size
        # Check all transformations
        for board_transform, move_transform, inverse_move_transform in transformations:
            transformed_board = board_transform(go.board)
            board_hash = self.hash_board(transformed_board)
            if board_hash in self.transposition_table:
                stored_move = self.transposition_table[board_hash]
                action = inverse_move_transform(stored_move[0], stored_move[1], N)
                return action

        # If no matching board found, proceed as before
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    possible_placements.append((i, j))

        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)

    def hash_board(self, board):
        return str(board)


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = RandomPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)