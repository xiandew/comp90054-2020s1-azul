from copy import deepcopy
import time

import numpy as np

from DQNAgent import DQNAgent

from advance_model import *
from utils import *


# features of the dataset are unknown and are going to get 
# selected by the Deep Q Network
# Here, we treat each round as a fresh new episode, when detecting an end round
# state(terminal state), we update the reward and output the weight backup file
# to the 'azul-qlearning' file, when entering the tournament, load the file and
# set epsilon value to minimum so that we do not consume excessive time and waste
# time explorating the steps.
class qLearningTrainer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.agent = DQNAgent()
        self.EPSILON = 1.0
        self.EPSILON_MIN = 0.01
        self.EPSILON_DECAY = 0.995

    def SelectMove(self, moves, game_state):

        # If this move is ending the round
        done = True if len(moves) == 1 else False

        # if not done:
        # before making moves, update Q values
        current_state = self.GetCurrentState(game_state)

        # choose the move for this step, using exploration-exploitation method,
        # Initially, the agent chooses all actions randomly
        # adjusting weights for features slowly

            # for each step in the episode
            # get the max q value, update the q value and neural network weights for current state
            # return if the reward is better than the previous one,
            # 

        if np.random.rand() <= self.EPSILON:
            selected_move = random.choice(moves)
        
        # select the move with highest q value from availale moves
        selected_move = np.argmax(self.agent.get_qs(current_state))

        # how to map actions according to each column vector.

        # calculate the reward from taking the move, using (future score - current score) 
        current_score = self.GetPlayerState(game_state).score
        new_state = MakeAMove(game_state, selected_move)
        new_score = self.GetPlayerState(new_state).score
        reward = new_score - current_score

        # add training resource to the replay memory deque for training later.
        agent.update_replay_memory((current_state, selected_move, reward, new_state, done))
        # train the model
        agent.train(done)

        agent.model.save(f'DQNModelVer1.model')
        # In the end, update the epsilon value
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY


    # modify the player state to be a (5 x 5 + 1 x 7) array
    # for input into the DQN model. 
    def GetCurrentState(self, game_state):
        player_state = self.GetPlayerState(game_state)
        player_state_copy = deepcopy(player_state)
        player_state_copy.grid_state.append(player_state_copy.floor)
        return player_state_copy.grid_state

    # Getting the player's current state for prediction
    def GetPlayerState(self, game_state):
        player_state = next(filter(lambda ps: ps.id == self.id, game_state.players))
    
    # Making move based on the selected action
    def MakeAMove(self, game_state, move):
        game_state_copy = deepcopy(game_state)
        return game_state_copy.ExecuteMove(self.id, move)

    def IsTerminalMove(self, game_state):
        game_state.