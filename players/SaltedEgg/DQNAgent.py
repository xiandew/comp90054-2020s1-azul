# Code with reference to online tutorial at python programming dot net
# https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
# part of the code was adapted to accommodate to our problem.

# Most of the comments are added by us.

# Our approach: In this implementation, we create a deep Q neural network with experience replay to enhance performance,
# We store past 'experience' in a deque and only for a certain amount of latest
# experience.
from collections import deque
from copy import deepcopy
import os

import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from advance_model import *
from utils import *

REPLAY_MEMORY_SIZE = 50000 # the maximum number of experience the memory deque can hold
MIN_REPLAY_MEMORY_SIZE = 1000 # the minimum number of experience when we carry out the training phase.
MINIBATCH_SIZE = 64 # train 64 replays everytime.
UPDATE_TARGET_EVERY = 5

LOAD_MODEL = None

class DQNAgent:
    def __init__(self):

        print('dqn agent')
        # Main model
        self.model = self.create_model()
        print('model created: ', self.model)
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
    
    # transition containing:
    # - current_state
    # - action
    # - reward
    # - new_state
    # - done
    # this is to add into the memory deque, and keep replaying the most recent
    # ones to update 
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state))[0]
    
    def train(self, terminal_state):
        if len(self.replay_memory) < MINIBATCH_SIZE:
            return
        
        # get some sample of (current_state, action, reward, new_state, done) to update later 
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Retrieve the states before action from the batch
        current_states = np.array([transition[0] for transition in minibatch])
        
        # Retrive the Q values before action from the batch
        current_qs_list = self.model.predict(current_states)
    
        # Retrieve the new states after peforming the actions from the batch
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

        
