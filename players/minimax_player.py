from copy import deepcopy

from advance_model import *
from utils import *


class myPlayer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.opponent_id = None

    def SelectMove(self, moves, game_state):
        self.opponent_id = next(filter(lambda ps: ps.id != self.id, game_state.players)).id
        return self.minimax(game_state, self.id, 2, float("-Inf"), float("Inf"))[1]

    # TODO Preferential ordering & Immediate pruning
    def minimax(self, game_state, player_id, depth, alpha, beta):
        if depth == 0:
            return self.evaluate(game_state), None
        # Maximising player
        if player_id == self.id:
            value = float("-Inf")
            best_move = None
            for move in get_available_moves(game_state, player_id):
                t = self.minimax(
                    result(game_state, player_id, move),
                    self.opponent_id,
                    depth - 1,
                    alpha,
                    beta
                )[0]
                if t > value:
                    value = t
                    best_move = move

                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move
        else:
            value = float("Inf")
            best_move = None
            for move in get_available_moves(game_state, player_id):
                t = self.minimax(
                    result(game_state, player_id, move),
                    self.id,
                    depth - 1,
                    alpha,
                    beta
                )[0]
                if t < value:
                    value = t
                    best_move = move

                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_move

    def evaluate(self, game_state):
        player_state = get_player_state(game_state, self.id)
        opponent_state = get_player_state(game_state, self.opponent_id)
        player_state.ScoreRound()
        opponent_state.ScoreRound()
        return player_state.score - opponent_state.score


def get_available_moves(game_state, player_id):
    return get_player_state(game_state, player_id).GetAvailableMoves(game_state)


def result(game_state, player_id, move):
    next_game_state = deepcopy(game_state)
    next_game_state.ExecuteMove(player_id, move)
    return next_game_state


def get_player_state(game_state, player_id):
    return next(filter(lambda ps: ps.id == player_id, game_state.players))
