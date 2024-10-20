# hyperparameter_tuning.py

import optuna
from host import GO
from random_player import RandomPlayer
from a_b_tunable import AlphaBetaPlayer_tunable
from alpha_best import AlphaBetaPlayer


def objective(trial):
    # Suggest values for the hyperparameters
    proximity_weight = trial.suggest_uniform('proximity_weight', 0, 10)
    wall_weight = trial.suggest_uniform('wall_weight', 0, 10)
    liberty_bonus = trial.suggest_uniform('liberty_bonus', 0, 10)
    eye_bonus = trial.suggest_uniform('eye_bonus', 0, 20)
    not_near_stone_penalty = trial.suggest_uniform('not_near_stone_penalty', -10, 0)

    # Create an AlphaBetaPlayer with these hyperparameters
    player = AlphaBetaPlayer_tunable(
        depth=2,  # Depth set to 2 for faster evaluation during tuning
        proximity_weight=proximity_weight,
        wall_weight=wall_weight,
        liberty_bonus=liberty_bonus,
        eye_bonus=eye_bonus,
        not_near_stone_penalty=not_near_stone_penalty
    )

    # Play games against RandomPlayer
    N = 5  # Board size
    num_games = 20  # Number of games per trial
    wins = 0

    for _ in range(num_games // 2):
        go = GO(N)
        go.verbose = False
        player1 = player
        player2 = AlphaBetaPlayer(depth = 3)
        result = go.play(player1, player2, verbose=False)
        if result == 1:
            wins += 1

    for _ in range(num_games // 2):
        go = GO(N)
        go.verbose = False
        player1 = AlphaBetaPlayer(depth = 3)
        player2 = player
        result = go.play(player1, player2, verbose=False)
        if result == 2:
            wins += 1

    win_rate = wins / num_games
    return win_rate

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1)  # Adjust n_trials as needed

    print('Best hyperparameters:')
    print(study.best_params)
    print('Best win rate:')
    print(study.best_value)

if __name__ == "__main__":
    main()