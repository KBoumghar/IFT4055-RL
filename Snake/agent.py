import torch
import numpy as np
import random
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.1

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # exploration
        self.gamma = 0.9 # discount rate
        self.memory = deque(MAX_MEMORY)
        pass

    def get_state(self):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_action(self, state):
        pass


def train():
    #Lists for graphs
    plot_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakegameAI()

    while True:
        #Get state we are in
        old_state = agent.get_state()

        #Action we should be doing with the state
        action = agent.get_action(old_state)

        #Perform the action
        


    #
    pass

if __name__ == '__main__':
    train()
