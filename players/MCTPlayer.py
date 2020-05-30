"""

"""
from copy import deepcopy
from advance_model import *
from utils import *
from collections import namedtuple
from random import choice
from monte_carlo_tree_search import MCTS, Node

_AZULB = namedtuple("AzulBoard", "game_state player_id opponent_id winner terminal")

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class AzulBoard(_AZULB, Node):

    #finding expanded state, returning a list of state
    def find_children(board):
        t = set()
        for move in get_available_moves(board.game_state, board.player_id):
            t.add(board.make_move(move))
        return t

    #Finding random move of the available move
    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        available_moves = get_available_moves(board.game_state, board.player_id)
        move = choice(available_moves)
        return board.make_move(move)

    #Returning reward if winner is us, and negative reward if winner is opponent
    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        player_state = get_player_state(board.game_state, board.player_id)
        opponent_state = get_player_state(board.game_state, board.opponent_id)
        player_score = player_state.ScoreRound()[0]
        opponent_score = opponent_state.ScoreRound()[0]
        if board.player_id == board.winner :
            return player_score  
        if board.player_id !=  board.winner:
            return -1*opponent_score
    
    def is_terminal(board):
        return board.terminal

    #For opponent to make their move and update the game board
    def make_move(board, move):
        nextstate = result(board.game_state, board.player_id, move)
        nextstate.lastmove = move
        winner_selected = board.winner_selection(nextstate)
        is_terminal = winner_selected is not None

        return AzulBoard(game_state = nextstate,player_id = board.opponent_id, opponent_id = board.player_id, winner=winner_selected, terminal=is_terminal)

    #Only return a winner if there is no more tiles left in game state.
    def winner_selection(board, newstate):

        if len(get_available_moves(newstate, board.opponent_id)) == 0:
            player_state = get_player_state(newstate, board.player_id)
            opponent_state = get_player_state(newstate, board.opponent_id)
            player_score = player_state.ScoreRound()[0]
            opponent_score = opponent_state.ScoreRound()[0]
            if player_score >= opponent_score:
                return board.player_id
            else:
                return board.opponent_id

        else:
            return None


def new_azul_board(gamestate, playerid, opponentid):
    gamestate.lastmove = None
    return AzulBoard(game_state = gamestate, player_id = playerid, opponent_id = opponentid, winner=None, terminal=False)

def get_available_moves(game_state, player_id):
    return get_player_state(game_state, player_id).GetAvailableMoves(game_state)

#Executing move and produced next state
def result(game_state, player_id, move):
    next_game_state = deepcopy(game_state)
    next_game_state.ExecuteMove(player_id, move)
    next_game_state.lastmove = move
    return next_game_state

def get_player_state(game_state, player_id):
    return next(filter(lambda ps: ps.id == player_id, game_state.players))


class myPlayer(AdvancePlayer):

    def __init__(self, _id):
        super().__init__(_id)
        self.opponent_id = None
        self.tree = MCTS()
        

    def SelectMove(self, moves, game_state):

        self.opponent_id = next(filter(lambda ps: ps.id != self.id, game_state.players)).id
        self.board = new_azul_board(game_state, self.id, self.opponent_id)

        for _ in range(1000):
           self.tree.do_rollout(self.board)
        best_board = self.tree.choose(self.board)

        return best_board.game_state.lastmove