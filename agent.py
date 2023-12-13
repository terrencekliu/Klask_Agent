# Inspired / refactored from https://github.com/patrickloeber/snake-ai-pytorch/tree/main

import random
import math
from collections import deque

import inquirer
import numpy as np
import torch

from helper import plot
from klask_constants import *
from klask_simulator import KlaskSimulator
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.0075

# Adaptive learning


class Agent:
    def __init__(self):
        # 17697
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(24, 256, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game, player):
        state = game.determine_agent_state()
        arr = np.array(state)

        return arr

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
        self.epsilon = 1000000 - self.n_games

        # force_x, force_y
        force = [0.0, 0.0]
        if random.randint(0, 100000) < self.epsilon:
            force[0] = random.uniform(-500, 500)
            force[1] = random.uniform(-500, 500)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            force[0] = prediction[0].item()
            force[1] = prediction[1].item()

        return force[0], force[1]


def train(mode: str):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    game = KlaskSimulator(render_mode=mode, target_fps=1000)
    game.reset()
    agent1 = Agent()
    agent2 = Agent()

    agent1.model.load("p1.pth")
    agent2.model.load("p2.pth")

    while True:
        # get old state
        state_old = agent1.get_state(game, game.PlayerPuck.P1)
        state_old2 = agent2.get_state(game, game.PlayerPuck.P2)

        # get move
        final_move = agent1.get_action(state_old)
        final_move2 = agent2.get_action(state_old2)

        # perform move and get new state
        frame, game_state, agent_states, reward, score = game.step(final_move, game.PlayerPuck.P1)
        frame2, game_state2, agent_states2, reward2, score2 = game.step(final_move2, game.PlayerPuck.P2)

        def __collide_with_ball_reward(player: game.PlayerPuck):
            puck_position = game.bodies[player.value].position
            ball_position = game.bodies["ball"].position

            # Calculate the distance between the centers of the puck and the ball
            distance = math.sqrt((puck_position.x - ball_position.x) ** 2 + (puck_position.y - ball_position.y) ** 2)

            # Check if the distance is less than the sum of their radii
            if distance < (KG_PUCK_RADIUS + KG_BALL_RADIUS) * game.length_scaler:
                return 0.1
            else:
                return 0

        def __is_ball_left_side(x_pos) -> bool:
            return x_pos < (KG_BOARD_WIDTH * game.length_scaler / 2)

        def __is_ball_right_side(x_pos) -> bool:
            return x_pos > (KG_BOARD_WIDTH * game.length_scaler / 2)

        def __ball_move_reward(player: game.PlayerPuck, game_states) -> float:
            x_pos = game_states[20]
            is_ball_move: bool = game_states[22] > 0.005 or game_states[23] > 0.005

            if player is game.PlayerPuck.P1 and __is_ball_right_side(x_pos) and is_ball_move:
                return 0.0125
            elif player is game.PlayerPuck.P1 and __is_ball_left_side(x_pos) and not is_ball_move:
                return -0.0125
            elif player is game.PlayerPuck.P2 and __is_ball_left_side(x_pos) and is_ball_move:
                return 0.0125
            elif player is game.PlayerPuck.P2 and __is_ball_right_side(x_pos) and not is_ball_move:
                return -0.0125
            else:
                return 0

        def __puck_move_reward(player: game.PlayerPuck, game_states) -> float:
            is_ball_move: bool = game_states[14] > 0.005 or game_states[15] > 0.005 or game_states[18] > 0.005 or game_states[19] > 0.005

            if not is_ball_move:
                return -0.0125
            return 0.0125

        def __ball_position_reward(player: game.PlayerPuck, game_states):
            x_pos_ball = game_states[20]
            x_pos_player = game_states[12] if player == game.PlayerPuck.P1 else game_states[16]

            if x_pos_ball < x_pos_player and player == game.PlayerPuck.P1:
                return -0.1
            elif x_pos_ball > x_pos_player and player == game.PlayerPuck.P2:
                return -0.1
            return 0

        def __calculate_reward(player: game.PlayerPuck, state: game.RewardState, game_states) -> float:
            reward = 0.0

            def __define_state_reward():
                if state == game.RewardState.PLAYING:
                    return 0
                elif state == game.RewardState.PUCK_IN_GOAL:
                    return 0
                elif state == game.RewardState.BALL_IN_GOAL:
                    return 1
                elif state == game.RewardState.MAGNET:
                    return 1
                elif state == game.RewardState.ONE_MAGNET:
                    return 1
                elif state == game.RewardState.SELF_PUCK_IN_GOAL:
                    return -1
                elif state == game.RewardState.SELF_BALL_IN_GOAL:
                    return -1
                elif state == game.RewardState.SELF_MAGNET:
                    return -1
                elif state == game.RewardState.SELF_ONE_MAG:
                    return -1
                elif state == game.RewardState.HIT_BALL:
                    return 0.1
                else:
                    return 0

            reward += __define_state_reward()
            # if player is game.PlayerPuck.P1 and reward != 0.0:
            #     print(reward)

            reward += __collide_with_ball_reward(player)
            reward += __ball_position_reward(player, game_states)
            # reward += self.__ball_stationary(player, game_states)
            reward += __ball_move_reward(player, game_states)

            return reward

        state_new = agent1.get_state(game, game.PlayerPuck.P1)
        state_new2 = agent2.get_state(game, game.PlayerPuck.P2)

        done_state = (game.RewardState.PLAYING, game.RewardState.ONE_MAGNET, game.RewardState.SELF_ONE_MAG)
        done = game_state not in done_state or game_state2 not in done_state

        calculated_reward = __calculate_reward(game.PlayerPuck.P1, game_state, agent_states)
        calculated_reward2 = __calculate_reward(game.PlayerPuck.P2, game_state2, agent_states2)

        if done:
            calculated_reward = -1 if calculated_reward2 > 0.0 else 1
            calculated_reward2 = -1 if calculated_reward > 0.0 else 1
            print("here is reward: ", end=": ")
            print(calculated_reward)

        # train short memory
        agent1.train_short_memory(state_old, final_move, calculated_reward, state_new, done)
        agent2.train_short_memory(state_old2, final_move2, calculated_reward2, state_new2, done)

        # remember
        agent1.remember(state_old, final_move, reward, state_new, done)
        agent2.remember(state_old2, final_move2, reward2, state_new2, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent1.n_games += 1
            agent1.train_long_memory()

            agent2.n_games += 1
            agent2.train_long_memory()

            # TODO: Fix graph to visualize game scores
            if score > record:
                record = score
                agent1.model.save("p1.pth")
                agent2.model.save("p2.pth")

            print('Game', agent1.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent1.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



# def train_helper(agent, game, player):
#     # get old state
#     state_old = agent.get_state(game)
#
#     # get move
#     final_move = agent.get_action(state_old)
#
#     # perform move and get new state
#     frame, game_state, agent_states, reward, score = game.step(final_move, player)
#     frame, game_state, agent_states, reward, score = game.step(final_move, player)
#
#     state_new = agent.get_state(game)
#     done_state = (game.RewardState.PLAYING, game.RewardState.ONE_MAGNET, game.RewardState.SELF_ONE_MAG)
#     done = game_state not in done_state
#     # train short memory
#     agent.train_short_memory(state_old, final_move, reward, state_new, done)
#
#     # remember
#     agent.remember(state_old, final_move, reward, state_new, done)
#
#     return frame, game_state, agent_states, reward, score, done


if __name__ == '__main__':
    questions = [
        inquirer.List(
            "headless",
            message="Headless?",
            choices=["y", "n"],
            default=["n"]
        ),
    ]

    answers = inquirer.prompt(questions)
    train("headless" if answers["headless"] == "y" else "human")
