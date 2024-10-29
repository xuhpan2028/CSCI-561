import optuna
from random_player import RandomPlayer
from player_greedy import GreedyPlayer
from player_aggressive import AggressivePlayer
from alpha_beta_player import AlphaBetaPlayer_
from host import GO

def objective(trial):
    # Sample the weights for each heuristic
    liberty_weight = trial.suggest_float('liberty_weight', 0, 10.0)
    threat_weight = trial.suggest_float('threat_weight', 0, 10.0)
    stone_weight = trial.suggest_float('stone_weight', 0, 10.0)
    center_weight = trial.suggest_float('center_weight', 0.01, 5.0)
    eye_weight = trial.suggest_float('eye_weight', 0, 10.0)
    aggression_weight = trial.suggest_float('aggression_weight', 0, 10.0)
    isolation_penalty = trial.suggest_float('isolation_penalty', 0, 10.0)

    # Create a tuned player with these weights
    weights = {
        'liberty_weight': liberty_weight,
        'threat_weight': threat_weight,
        'stone_weight': stone_weight,
        'center_weight': center_weight,
        'eye_weight': eye_weight,
        'aggression_weight': aggression_weight,
        'isolation_penalty': isolation_penalty
    }
    
    player = AlphaBetaPlayer_(depth=3, weights=weights)

    # Simulate games and return the number of wins
    opponents = [
        # RandomPlayer(),
        GreedyPlayer(),
        AggressivePlayer()
    ]
    

    N = 5  # Board size
    wins = 0
    
    for _ in range(50):
        for opponent in opponents:

            # Play as white
            go = GO(N)
            go.verbose = False
            player1 = opponent  # Opponent as black
            player2 = player  # Tuned player as white
            result = go.play(player1, player2, verbose=False)
            if result == 2:
                wins += 1

    return wins

# Create the study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5000, n_jobs=-1)