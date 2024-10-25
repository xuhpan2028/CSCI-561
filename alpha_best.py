import math
from read import readInput
from write import writeOutput
from host import GO

class AlphaBetaPlayer:
    def __init__(self, depth=3, 
                 proximity_weight=9.069364089954512,
                 wall_weight=4.320619406707552,
                 liberty_bonus=9.345235615125652,
                 eye_bonus=4.18670084338185,
                 not_near_stone_penalty=-2.6480232913854875,
                 capturing_stones_weight=80.92947316054017,
                 reduce_opponent_liberties_weight=8.522578859336393,
                 threatened_groups_weight=7.348797614269035):
        self.type = 'alpha-beta'
        self.depth = depth
        self.transposition_table = {}  # Store evaluated board states

        # Hyperparameters for tuning
        self.proximity_weight = proximity_weight
        self.wall_weight = wall_weight
        self.liberty_bonus = liberty_bonus
        self.eye_bonus = eye_bonus
        self.not_near_stone_penalty = not_near_stone_penalty
        self.capturing_stones_weight = capturing_stones_weight
        self.reduce_opponent_liberties_weight = reduce_opponent_liberties_weight
        self.threatened_groups_weight = threatened_groups_weight

    def get_input(self, go, piece_type):
        best_move = None
        best_score = -math.inf
        
        # Check if center is available and place there if possible
        center = (go.size // 2, go.size // 2)
        if go.valid_place_check(center[0], center[1], piece_type, test_check=True):
            return center  # Return the center position if it's available
        # Order moves based on Go tactics
        possible_moves = self.get_ordered_moves(go, piece_type)


        # First, check if any move can capture opponent stones
        capturing_moves = []
        for move in possible_moves:
            i, j = move
            test_go = go.copy_board()
            if not test_go.place_chess(i, j, piece_type):
                continue
            dead_stones = test_go.remove_died_pieces(3 - piece_type)
            if dead_stones:
                capturing_moves.append((i, j, len(dead_stones)))

        if capturing_moves:
            # If there are capturing moves, pick the one that captures the most stones
            capturing_moves.sort(key=lambda x: x[2], reverse=True)  # Sort by number of stones captured
            best_move = (capturing_moves[0][0], capturing_moves[0][1])
            return best_move  # Return the capturing move immediately


        # **Modify the depth based on board occupancy**
        empty_spots = sum(row.count(0) for row in go.board)
        if empty_spots <= 18:
            depth = 4
        elif empty_spots <= 14:
            depth = 5
        elif empty_spots <= 12:
            depth = 6
        elif empty_spots <= 9:
            depth = 9
        else:
            depth = self.depth  # Use the default depth


        for move in possible_moves:
            i, j = move
            test_go = go.copy_board()
            if not test_go.place_chess(i, j, piece_type):
                continue
            score = self.alpha_beta(test_go, depth - 1, -math.inf, math.inf, False, piece_type)

            if score > best_score:
                best_score = score
                best_move = (i, j)

        if best_move is None:
            return "PASS"
        else:
            return best_move

    def alpha_beta(self, go, depth, alpha, beta, maximizing_player, piece_type):
        """Alpha-beta pruning algorithm with transposition table for efficiency."""
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
                if not test_go.place_chess(i, j, piece_type):
                    continue
                eval = self.alpha_beta(test_go, depth - 1, alpha, beta, False, piece_type)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.transposition_table[board_hash] = max_eval
            return max_eval
        else:
            min_eval = math.inf
            opponent_type = 3 - piece_type
            possible_moves = self.get_ordered_moves(go, opponent_type)
            for move in possible_moves:
                i, j = move
                test_go = go.copy_board()
                if not test_go.place_chess(i, j, opponent_type):
                    continue
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
        discount = 0.1 if piece_type == 2 else 1.0  # Apply discount for White

        distance_to_center = abs(i - center) + abs(j - center)
        score += (go.size - distance_to_center) * self.proximity_weight  # Proximity to center

        # Get wall sizes (horizontal and vertical)
        horizontal_length, vertical_length = self.forms_wall(go, i, j, piece_type)
        
        # Add score based on the size of the wall
        if horizontal_length > 1:
            score += horizontal_length * self.wall_weight
        if vertical_length > 1:
            score += vertical_length * self.wall_weight

        # Favor moves that increase liberties
        if go.find_liberty(i, j):
            score += self.liberty_bonus

        # Favor moves that create or defend eyes
        if self.creates_eye(go, i, j, piece_type):
            score += self.eye_bonus

        # Penalize moves that have no impact
        if not self.is_near_stone(go, i, j):
            score += self.not_near_stone_penalty

        

        # Simulate placing the stone and count captured opponent stones
        test_go = go.copy_board()
        if test_go.place_chess(i, j, piece_type):
            dead_stones = test_go.remove_died_pieces(3 - piece_type)
            score += len(dead_stones) * self.capturing_stones_weight * discount

            # Extra aggression: reduce opponent liberties
            liberties_reduction = self.reduce_opponent_liberties(go, i, j, piece_type)
            score += liberties_reduction * self.reduce_opponent_liberties_weight * discount

            # Threaten opponent groups
            threatened_groups = self.count_threatened_groups(go, i, j, piece_type)
            score += threatened_groups * self.threatened_groups_weight * discount

        return score

    def creates_eye(self, go, i, j, piece_type):
        """Determine if placing a piece at (i, j) would create an eye."""
        neighbors = go.detect_neighbor(i, j)
        for ni, nj in neighbors:
            if go.board[ni][nj] != piece_type:
                return False
        return True

    def forms_wall(self, go, i, j, piece_type):
        """Calculate the length of horizontal and vertical walls around the move."""
        horizontal_length = 1
        # Left
        j2 = j - 1
        while j2 >= 0 and go.board[i][j2] == piece_type:
            horizontal_length += 1
            j2 -= 1
        # Right
        j2 = j + 1
        while j2 < go.size and go.board[i][j2] == piece_type:
            horizontal_length += 1
            j2 += 1

        vertical_length = 1
        # Up
        i2 = i - 1
        while i2 >= 0 and go.board[i2][j] == piece_type:
            vertical_length += 1
            i2 -= 1
        # Down
        i2 = i + 1
        while i2 < go.size and go.board[i2][j] == piece_type:
            vertical_length += 1
            i2 += 1

        return horizontal_length, vertical_length

    def is_near_stone(self, go, i, j):
        """Check if the position is near another stone on the board."""
        neighbors = go.detect_neighbor(i, j)
        for ni, nj in neighbors:
            if go.board[ni][nj] != 0:
                return True
        return False

    def hash_board(self, board):
        return str(board)  # Simple hash based on board's string representation

    def evaluate(self, go, piece_type):
        # Basic score: difference in stone count
        my_stones = go.score(piece_type)
        opponent_stones = go.score(3 - piece_type)
        score = my_stones - opponent_stones

        # Additional heuristic: count liberties
        my_liberties = self.count_total_liberties(go, piece_type)
        opponent_liberties = self.count_total_liberties(go, 3 - piece_type)
        liberties = my_liberties - opponent_liberties

        if piece_type == 1:
            # For black, be aggressive
            score += liberties * 0.5
            # Penalize opponent stones more heavily
            score -= opponent_stones * 1.5
        else:
            # For white, play normal
            score += liberties * 0.5

        return score

    def count_total_liberties(self, go, piece_type):
        total_liberties = 0
        visited = set()
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type and (i, j) not in visited:
                    group = go.ally_dfs(i, j)
                    visited.update(group)
                    liberties = self.count_group_liberties(go, group)
                    total_liberties += liberties
        return total_liberties

    def count_group_liberties(self, go, group):
        """Counts the number of liberties for a group of stones."""
        liberties = set()
        for i, j in group:
            neighbors = go.detect_neighbor(i, j)
            for ni, nj in neighbors:
                if go.board[ni][nj] == 0:
                    liberties.add((ni, nj))
        return len(liberties)

    def reduce_opponent_liberties(self, go, i, j, piece_type):
        """Calculate the reduction in opponent's liberties after placing a stone at (i, j)."""
        test_go = go.copy_board()
        if test_go.place_chess(i, j, piece_type):
            opponent_type = 3 - piece_type
            opponent_liberties_before = self.count_total_liberties(go, opponent_type)
            opponent_liberties_after = self.count_total_liberties(test_go, opponent_type)
            return opponent_liberties_before - opponent_liberties_after
        else:
            return 0

    def count_threatened_groups(self, go, i, j, piece_type):
        """Count the number of opponent groups that are put in atari (one liberty left)."""
        test_go = go.copy_board()
        if test_go.place_chess(i, j, piece_type):
            opponent_type = 3 - piece_type
            threatened_groups = 0
            visited = set()
            for x in range(test_go.size):
                for y in range(test_go.size):
                    if test_go.board[x][y] == opponent_type and (x, y) not in visited:
                        group = test_go.ally_dfs(x, y)
                        visited.update(group)
                        liberties = self.count_group_liberties(test_go, group)
                        if liberties == 1:
                            threatened_groups += 1
            return threatened_groups
        else:
            return 0
    def detect_wall_length(go, piece_type):
        """
        Detect the length of the wall (horizontally or vertically connected stones) for a given piece type.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: Maximum length of the wall for the given piece type.
        """
        max_wall_length = 0
        board_size = go.size

        # Check horizontal walls
        for i in range(board_size):
            current_length = 0
            for j in range(board_size):
                if go.board[i][j] == piece_type:
                    current_length += 1
                    max_wall_length = max(max_wall_length, current_length)
                else:
                    current_length = 0  # Reset length if the stone is not part of the wall

        # Check vertical walls
        for j in range(board_size):
            current_length = 0
            for i in range(board_size):
                if go.board[i][j] == piece_type:
                    current_length += 1
                    max_wall_length = max(max_wall_length, current_length)
                else:
                    current_length = 0  # Reset length if the stone is not part of the wall

        return max_wall_length
        
if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = AlphaBetaPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)