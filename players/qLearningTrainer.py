from copy import deepcopy

import numpy as np

from DQNAgent import DQNAgent

from advance_model import *
from utils import *

class qLearningTrainer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.agent = DQNAgent()
        self.EPSILON = 1.0
        self.EPSILON_MIN = 0.01
        self.EPSILON_DECAY = 0.995

    def SelectMove(self, moves, game_state):
        # before making moves, update Q values
        current_state = self.GetCurrentState(game_state)
        if np.random.rand() <= self.EPSILON:
            return random.choice(moves)
        
        action = np.argmax(self.agent.get_qs(current_state))
        
    def GetCurrentState(self, game_state):
        player_state = next(filter(lambda ps: ps.id == self.id, game_state.players))
        player_state_copy = deepcopy(player_state)
        player_state_copy.grid_state.append(player_state_copy.floor)
        return player_state_copy.grid_state
