from collections import deque
from copy import deepcopy

import numpy as np

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

from advance_model import *
from utils import *

class qPlayer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.state_size = (2 ** 5 ** 5) * (2 ** 7)
        # we calculate current player's number of states, 2**5 ** 5 * 2 ** 7
        self.action_size = 5 * 5 * 4 * (5 * 15) # 7500 possible actions.
        # 5 types of colours from centre, ----- 5 actions
        # 5 types of colours from factory ----- 5 * 5
        # 5 factories
        self.weight_backup = "azul_qlearning"
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.brain = self._build_model()

    def _build_model(self):
        # Neural network for deepQ learning model.
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile("self.weight_backup"):
            model.load_weights(self.weight_backup)
            self.epsilon = self.epsilon_min
            return model
    
    def save_model(self):
        self.brain.save(self.weight_backup)

    def SelectMove(self, moves, game_state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            # get available moves instead
        act_values = self.brain.predict(game_state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, sample_batch_size = 32):
        if len(self.memory) < sample_batch_size:
            return
        
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                # update the q value of current state with certain action.
                target = reward + self.gamma * np.argmax(self.brain.predict(next_state)[0])
            target_to_update = self.brain.predict(state)
            target_to_update[0][action] = target
            self.brain.fit(state, target_to_update, epochs = 1, verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def action_result(game_state, player_id, move):
    game_state_copy = deepcopy(game_state)
    game_state_copy.ExecuteMove(player_id, move)
    return game_state_copy

        
        

    



