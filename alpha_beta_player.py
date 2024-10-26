import math
from read import readInput
from write import writeOutput
from host import GO

class AlphaBetaPlayer_:
    def __init__(self, depth=3, weights=None):
        self.type = 'alpha-beta'
        self.depth = depth
        self.transposition_table = {}
        self.weights = weights or {'liberty_weight': 1.9534765905084095, 'threat_weight': 6.149773443181987, 'stone_weight': 4.556323688111932, 'center_weight': 3.293051824747559, 'eye_weight': 2.8763923920290906, 'aggression_weight': 2.6616146706417156, 'isolation_penalty': 2.980611980470317}

    def get_input(self, go, piece_type):
        best_move = None
        best_score = -math.inf
        
        # Check if center is available and place there if possible
        center = (go.size // 2, go.size // 2)
        if go.valid_place_check(center[0], center[1], piece_type, test_check=True):
            return center  # Return the center position if it's available
        # Order moves based on Go tactics
        possible_moves = self.get_moves(go, piece_type)


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
            possible_moves = self.get_moves(go, piece_type)
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
            possible_moves = self.get_moves(go, opponent_type)
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

    def get_moves(self, go, piece_type):
        moves = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    moves.append((i, j))
        return moves  

    def hash_board(self, board):
        return str(board)  # Simple hash based on board's string representation

    def evaluate(self, go, piece_type):
        # Base score: difference in stone count
        my_stones = go.score(piece_type)
        opponent_stones = go.score(3 - piece_type)
        score = (my_stones - opponent_stones) * self.weights['stone_weight']

        # Count liberties
        my_liberties = self.count_total_liberties(go, piece_type)
        opponent_liberties = self.count_total_liberties(go, 3 - piece_type)
        liberties = my_liberties - opponent_liberties
        score += liberties * self.weights['liberty_weight']

        # Prefer central positions
        center = (go.size // 2, go.size // 2)
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type:
                    distance_to_center = abs(i - center[0]) + abs(j - center[1])
                    score -= distance_to_center * self.weights['center_weight']

        # Eye counting
        my_eyes = self.count_eyes(go, piece_type)
        opponent_eyes = self.count_eyes(go, 3 - piece_type)
        score += (my_eyes - opponent_eyes) * self.weights['eye_weight']

        # Adjust based on aggression or defense
        if piece_type == 1:  # Black (aggressive)
            threatened_groups = self.count_threatened_groups(go, 3 - piece_type)
            score += threatened_groups * self.weights['aggression_weight']

        # Penalize isolated moves
        isolation_penalty = self.calculate_isolation_penalty(go, piece_type)
        score += isolation_penalty * self.weights['isolation_penalty']

        score -= self.count_threatened_groups(go, piece_type) * self.weights['threat_weight']
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
    
    def count_eyes(self, go, piece_type):
        eyes = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type:
                    neighbors = go.detect_neighbor(i, j)
                    # An eye is a position where all neighbors are of the same type
                    if all(go.board[ni][nj] == piece_type for ni, nj in neighbors):
                        eyes += 1
        return eyes
    
    def count_threatened_groups(self, go, opponent_type):
        threatened_groups = 0
        visited = set()
        for i in range(go.size):
            for j in range(go.size): 
                if go.board[i][j] == opponent_type and (i, j) not in visited:
                    group = go.ally_dfs(i, j)
                    visited.update(group)
                    liberties = self.count_group_liberties(go, group)
                    if liberties == 1:
                        threatened_groups += 1
        return threatened_groups
    

    def calculate_isolation_penalty(self, go, piece_type):
        """
        Calculate isolation penalty based on the proximity of stones to allies.
        Penalizes stones that have fewer allied neighbors.
        """
        isolation_penalty = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type:
                    # Check if stone has allies nearby
                    neighbors = go.detect_neighbor(i, j)
                    adjacent_allies = sum(1 for (ni, nj) in neighbors if go.board[ni][nj] == piece_type)
                    if adjacent_allies == 0:  # Penalize if no allies are adjacent
                        isolation_penalty -= 1
        return isolation_penalty
        
if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = AlphaBetaPlayer_()
    action = player.get_input(go, piece_type)
    writeOutput(action)