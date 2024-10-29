import random
import time
import sys
from read import readInput
from write import writeOutput

from host import GO

class MCTSNode:
    def __init__(self, go, parent=None, move=None, piece_type=1):
        self.go = go
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = self.get_legal_moves(piece_type)
        self.piece_type = piece_type

    def get_legal_moves(self, piece_type):
        moves = []
        for i in range(self.go.size):
            for j in range(self.go.size):
                if self.go.valid_place_check(i, j, piece_type, test_check=True):
                    moves.append((i, j))
        if not moves:
            moves = ["PASS"]
        return moves

    def UCT_select_child(self):
        # Use Upper Confidence Bound applied to trees formula
        import math
        log_visits = math.log(self.visits)
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            exploration = math.sqrt(2 * log_visits / child.visits) if child.visits > 0 else float('inf')
            uct_score = win_rate + exploration
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    def add_child(self, move, go, piece_type):
        child_node = MCTSNode(go, parent=self, move=move, piece_type=piece_type)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        if result == self.piece_type:
            self.wins += 1
        elif result == 0:
            self.wins += 0.5  # For tie
        # else: do nothing (loss)

class MCTSPlayer:
    def __init__(self, calculation_time=5):
        self.type = 'mcts'
        self.calculation_time = calculation_time  # Time in seconds to run MCTS

    def get_input(self, go, piece_type):
        '''
        Get one input based on MCTS strategy.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input or "PASS".
        '''
        import time
        start_time = time.time()
        root_node = MCTSNode(go.copy_board(), piece_type=piece_type)

        while time.time() - start_time < self.calculation_time:
            node = root_node
            go_copy = go.copy_board()

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.UCT_select_child()
                if node.move != "PASS":
                    go_copy.place_chess(node.move[0], node.move[1], node.piece_type)
                    go_copy.remove_died_pieces(3 - node.piece_type)
                else:
                    go_copy.previous_board = go_copy.board

            # Expansion
            if node.untried_moves != []:
                m = random.choice(node.untried_moves)
                go_copy_sim = go_copy.copy_board()
                if m != "PASS":
                    go_copy_sim.place_chess(m[0], m[1], node.piece_type)
                    go_copy_sim.remove_died_pieces(3 - node.piece_type)
                else:
                    go_copy_sim.previous_board = go_copy_sim.board
                node = node.add_child(m, go_copy_sim, 3 - node.piece_type)

            # Simulation
            result = self.simulate(go_copy, 3 - node.piece_type)

            # Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent

        # Choose the move with the highest visit count
        best_move = None
        max_visits = -1
        for child in root_node.children:
            if child.visits > max_visits:
                max_visits = child.visits
                best_move = child.move

        if best_move == "PASS":
            return "PASS"
        else:
            return best_move

    def simulate(self, go, piece_type):
        '''
        Simulate a random playout from the current state.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: The result of the game: 1('X') wins, 2('O') wins, or 0(tie).
        '''
        max_simulation_moves = go.size * go.size * 2  # Arbitrary limit to prevent infinite games
        simulation_moves = 0

        while not go.game_end(piece_type) and simulation_moves < max_simulation_moves:
            legal_moves = []
            for i in range(go.size):
                for j in range(go.size):
                    if go.valid_place_check(i, j, piece_type, test_check=True):
                        legal_moves.append((i, j))
            if legal_moves:
                move = random.choice(legal_moves)
                go.place_chess(move[0], move[1], piece_type)
                go.remove_died_pieces(3 - piece_type)
            else:
                # Pass if no legal moves
                go.previous_board = go.board
            piece_type = 3 - piece_type
            simulation_moves += 1

        result = go.judge_winner()
        return result

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MCTSPlayer(calculation_time=1)  # Adjust calculation time as needed
    action = player.get_input(go, piece_type)
    writeOutput(action)