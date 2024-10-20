from read import readInput
from write import writeOutput
from host import GO
import math

class AlphaBetaPlayer_tunable:
    def __init__(self, depth=3, proximity_weight=5, wall_weight=3,
                 liberty_bonus=5, eye_bonus=15, not_near_stone_penalty=-5):
        self.type = 'alpha-beta'
        self.depth = depth
        self.transposition_table = {}  # Store evaluated board states

        # Hyperparameters for evaluate_move
        self.proximity_weight = proximity_weight
        self.wall_weight = wall_weight
        self.liberty_bonus = liberty_bonus
        self.eye_bonus = eye_bonus
        self.not_near_stone_penalty = not_near_stone_penalty

    def get_input(self, go, piece_type):
        best_move = None
        best_score = -math.inf

        # Order moves based on Go tactics
        possible_moves = self.get_ordered_moves(go, piece_type)
        
        for move in possible_moves:
            i, j = move
            test_go = go.copy_board()
            test_go.place_chess(i, j, piece_type)
            score = self.alpha_beta(test_go, self.depth, -math.inf, math.inf, False, piece_type)
            
            if score > best_score:
                best_score = score
                best_move = (i, j)

        if best_move is None:
            return "PASS"
        else:
            return best_move

    def alpha_beta(self, go, depth, alpha, beta, maximizing_player, piece_type):
        '''
        Alpha-beta pruning algorithm with transposition table for efficiency.
        '''
        # Check if the board state is already evaluated
        board_hash = self.hash_board(go.board)
        if board_hash in self.transposition_table:
            return self.transposition_table[board_hash]

        if depth == 0 or go.game_end(piece_type):
            score = self.evaluate(go, piece_type)
            self.transposition_table[board_hash] = score
            return score

        if maximizing_player:
            max_eval = -math.inf
            possible_moves = self.get_ordered_moves(go, piece_type)
            for move in possible_moves:
                i, j = move
                test_go = go.copy_board()
                test_go.place_chess(i, j, piece_type)
                eval = self.alpha_beta(test_go, depth - 1, alpha, beta, False, piece_type)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.transposition_table[board_hash] = max_eval
            return max_eval
        else:
            min_eval = math.inf
            possible_moves = self.get_ordered_moves(go, 3 - piece_type)
            for move in possible_moves:
                i, j = move
                test_go = go.copy_board()
                test_go.place_chess(i, j, 3 - piece_type)
                eval = self.alpha_beta(test_go, depth - 1, alpha, beta, True, piece_type)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.transposition_table[board_hash] = min_eval
            return min_eval

    def get_ordered_moves(self, go, piece_type):
        moves = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    move_score = self.evaluate_move(go, i, j, piece_type)
                    moves.append((i, j, move_score))

        # Sort moves by score (higher score first)
        moves.sort(key=lambda x: x[2], reverse=True)
        return [(i, j) for i, j, score in moves]

    def evaluate_move(self, go, i, j, piece_type):
        score = 0
        center = go.size // 2

        distance_to_center = abs(i - center) + abs(j - center)
        score += (go.size - distance_to_center) * self.proximity_weight  # Score more for proximity to center

        # Get wall sizes (horizontal and vertical)
        horizontal_length, vertical_length = self.forms_wall(go, i, j, piece_type)
        
        # Add score based on the size of the wall
        if horizontal_length > 1:
            score += horizontal_length * self.wall_weight  # Horizontal wall
        if vertical_length > 1:
            score += vertical_length * self.wall_weight  # Vertical wall

        # Favor moves that increase liberties
        if go.find_liberty(i, j):
            score += self.liberty_bonus  # Moves that increase liberties are good

        # Favor moves that create or defend eyes
        if self.creates_eye(go, i, j, piece_type):
            score += self.eye_bonus

        # Penalize moves that are not near other stones
        if not self.is_near_stone(go, i, j):
            score += self.not_near_stone_penalty

        return score

    def creates_eye(self, go, i, j, piece_type):
        '''
        Determine if placing a piece at (i, j) would create an eye.
        '''
        neighbors = go.detect_neighbor(i, j)
        empty_neighbors = 0
        for neighbor in neighbors:
            ni, nj = neighbor
            if go.board[ni][nj] == 0:
                empty_neighbors += 1
        # A simple check for an eye: there must be at least 3 empty neighbors
        return empty_neighbors >= 3
    
    def forms_wall(self, go, i, j, piece_type):
        """
        Calculate the length of horizontal and vertical walls around the move.
        """
        # Calculate horizontal wall size
        horizontal_length = 1  # Start with 1 because the current stone is part of the wall
        # Count to the left
        for j2 in range(j - 1, -1, -1):
            if go.board[i][j2] == piece_type:
                horizontal_length += 1
            else:
                break
        # Count to the right
        for j2 in range(j + 1, go.size):
            if go.board[i][j2] == piece_type:
                horizontal_length += 1
            else:
                break

        # Calculate vertical wall size
        vertical_length = 1  # Start with 1 because the current stone is part of the wall
        # Count upwards
        for i2 in range(i - 1, -1, -1):
            if go.board[i2][j] == piece_type:
                vertical_length += 1
            else:
                break
        # Count downwards
        for i2 in range(i + 1, go.size):
            if go.board[i2][j] == piece_type:
                vertical_length += 1
            else:
                break

        return horizontal_length, vertical_length

    def is_near_stone(self, go, i, j):
        '''
        Check if the position is near another stone on the board.
        '''
        neighbors = go.detect_neighbor(i, j)
        for neighbor in neighbors:
            if go.board[neighbor[0]][neighbor[1]] != 0:
                return True
        return False

    def hash_board(self, board):
        return str(board)  # Simple hash based on board's string representation

    def evaluate(self, go, piece_type):
        # Basic score: difference in stone count
        score = go.score(piece_type) - go.score(3 - piece_type)

        # Additional heuristic: count liberties
        liberties = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type and go.find_liberty(i, j):
                    liberties += 1
                elif go.board[i][j] == 3 - piece_type and go.find_liberty(i, j):
                    liberties -= 1

        return score + liberties * 0.5  # Weight liberties lower than raw stone count


# if __name__ == "__main__":
#     N = 5
#     piece_type, previous_board, board = readInput(N)
#     go = GO(N)
#     go.set_board(piece_type, previous_board, board)
#     player = AlphaBetaPlayer_tunable(depth=3)  # Set depth to 3 or more for deep search
#     action = player.get_input(go, piece_type)
#     writeOutput(action)