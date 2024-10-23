import random
import sys
from read import readInput
from write import writeOutput

from host import GO

class AggressivePlayer():
    def __init__(self):
        self.type = 'aggressive'

    def get_input(self, go, piece_type):
        '''
        Get one input based on an aggressive strategy.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input or "PASS".
        '''
        max_aggression = -float('inf')
        best_moves = []

        opponent_type = 3 - piece_type

        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    # Copy the board for simulation
                    test_go = go.copy_board()
                    test_go.place_chess(i, j, piece_type)
                    # Remove dead opponent stones
                    dead_stones = test_go.remove_died_pieces(opponent_type)
                    num_captured = len(dead_stones)

                    # Calculate aggression score
                    aggression_score = num_captured * 10  # Prioritize capturing stones

                    # Reduce opponent liberties
                    opponent_liberties = self.count_opponent_liberties(test_go, opponent_type)
                    aggression_score += (self.count_opponent_liberties(go, opponent_type) - opponent_liberties)

                    # Threaten opponent groups
                    threatened_groups = self.count_threatened_groups(test_go, opponent_type)
                    aggression_score += threatened_groups * 5

                    # Prefer moves closer to the center
                    distance_to_center = abs(i - go.size // 2) + abs(j - go.size // 2)
                    aggression_score -= distance_to_center * 0.1  # Slightly prefer center positions

                    if aggression_score > max_aggression:
                        max_aggression = aggression_score
                        best_moves = [(i, j)]
                    elif aggression_score == max_aggression:
                        best_moves.append((i, j))

        if best_moves:
            return random.choice(best_moves)
        else:
            # If no aggressive moves are possible, return "PASS"
            return "PASS"

    def count_opponent_liberties(self, go, opponent_type):
        total_liberties = 0
        visited = set()
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == opponent_type and (i, j) not in visited:
                    group = go.ally_dfs(i, j)
                    visited.update(group)
                    liberties = self.count_group_liberties(go, group)
                    total_liberties += liberties
        return total_liberties

    def count_group_liberties(self, go, group):
        '''
        Count the number of liberties for a group of stones.

        :param go: Go instance.
        :param group: List of positions in the group.
        :return: Number of liberties for the group.
        '''
        liberties = set()
        for (i, j) in group:
            neighbors = go.detect_neighbor(i, j)
            for (x, y) in neighbors:
                if go.board[x][y] == 0:
                    liberties.add((x, y))
        return len(liberties)

    def count_threatened_groups(self, go, opponent_type):
        '''
        Count the number of opponent groups that have only one liberty (in atari).

        :param go: Go instance.
        :param opponent_type: 1('X') or 2('O').
        :return: Number of opponent groups in atari.
        '''
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

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = AggressivePlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)
