from collections import deque
from copy import deepcopy
import os

import numpy as np

from tensorflow.keras import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from advance_model import *
from utils import *

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64

class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_model_update_counter = 0

        # Some constants
        self.ACTION_SIZE = 7500
        # 5 * 5 * 4 * (5 * 15)
        self.LEARNING_RATE = 0.1
        self.GAMMA = 0.9


    def create_model(self):
        if LOAD_MODEL is not None:
            model = load_model(LOAD_MODEL)
            self.EPSILON = self.EPSILON_MIN

            return model

        model = Sequential()
        # the input shape here would be the player state:
        # [[0,0,0,0,0],
        #  [0,0,0,0,0],
        #  [0,0,0,0,0],
        #  [0,0,0,0,0]
        #  [0,0,0,0,0]]
        # together with a [0,0,0,0,0,0,0] floor state
        model.add(Dense(256, input_shape=(6,), activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.LEARNING_RATE))

        # if os.path.isfile(self.trained_model):
        #     model = load
        #     self.EPSILON = self.EPSILON_MIN
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state))[0]
    
    def train(self, terminal_state, step):
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
    
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = [] # states
        y = [] # Q values corresponding to actions

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.GAMMA * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0)

        if terminal_state:
            self.target_model_update_counter += 1
        
        if self.target_model_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_model_update_counter = 0

        
