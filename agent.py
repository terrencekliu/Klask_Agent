# Inspired / refactored from https://github.com/patrickloeber/snake-ai-pytorch/tree/main

import torch
import random
import numpy as np
from collections import deque

import model
from klask_simulator import KlaskSimulator
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.05

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(24, 480, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = game.determine_agent_state()
        return np.array(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 1000 - self.n_games

        # force_x, force_y
        force = [0.0, 0.0]
        if random.randint(0, 400) < self.epsilon:
            force[0] = random.uniform(-500, 500)
            force[1] = random.uniform(-500, 500)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            force[0] = prediction[0].item()
            force[1] = prediction[1].item()

        return force[0], force[1]


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    game = KlaskSimulator(render_mode="human", target_fps=500)
    game.reset()
    agent1 = Agent()
    agent2 = Agent()

    while True:
        frame, game_state, agent_states, reward, score, done = train_helper(agent1, game, game.PlayerPuck.P1)
        train_helper(agent2, game, game.PlayerPuck.P2)

        if done:
            # train long memory, plot result
            game.reset()
            agent1.n_games += 1
            agent1.train_long_memory()

            agent2.n_games += 1
            agent2.train_long_memory()

            if score > record:
                record = score
                agent1.model.save()
                agent2.model.save()

            print('Game', agent1.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent1.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def train_helper(agent, game, player):
    # get old state
    state_old = agent.get_state(game)

    # get move
    final_move = agent.get_action(state_old)

    # perform move and get new state
    frame, game_state, agent_states, reward, score = game.step(final_move, player)
    state_new = agent.get_state(game)
    done = game_state is not game.RewardState.PLAYING

    # train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, done)

    # remember
    agent.remember(state_old, final_move, reward, state_new, done)

    return frame, game_state, agent_states, reward, score, done

if __name__ == '__main__':
    train()