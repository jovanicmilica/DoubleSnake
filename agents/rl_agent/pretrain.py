"""
Predtreniranje replay buffera koristeci heuristic vs heuristic igre.
"""

import numpy as np

from game.game import Game
from game.state_utils import STATE_SIZE, get_state
from agents.heuristic_agent.heuristic_agent import HeuristicAgent
from agents.rl_agent.replay_buffer import ReplayBuffer
from agents.rl_agent.snake_env import ACTIONS
from config.config import (
    HEURISTIC_BUFFER_SIZE,
    HEURISTIC_PRETRAIN_EPISODES,
    HEURISTIC_ACTION_RANDOM_PROB,
)


def pretrain_buffer(
    num_episodes=HEURISTIC_PRETRAIN_EPISODES,
    buffer_size=HEURISTIC_BUFFER_SIZE,
    random_action_prob=HEURISTIC_ACTION_RANDOM_PROB,
):
    """
    Pokrece heuristic agenta kao obje zmije i puni poseban (heuristic) buffer
    tranzicijama iz perspektive snake1 (iste obs kakve DQN vidi tokom treninga).

    Ovaj buffer se ne mesa sa self-play bufferom DQN agenta - koristi se
    samo kao odvojen izvor iskustava za train_step dok DQN puni svoj buffer
    """
    buffer = ReplayBuffer(buffer_size)

    episode = 0
    while len(buffer) < buffer_size and episode < num_episodes:
        game = Game()
        h1 = HeuristicAgent(game.snake1, game.snake2, game.board, random_action_prob)
        h2 = HeuristicAgent(game.snake2, game.snake1, game.board, random_action_prob)

        done = False
        while not done:
            obs = get_state(game.snake1, game.snake2, game.board)

            dir1 = h1.get_action()
            dir2 = h2.get_action()
            action = ACTIONS.index(dir1)

            reward1, _, done = game.step(dir1, dir2)

            next_obs = (
                get_state(game.snake1, game.snake2, game.board)
                if game.snake1.alive
                else np.zeros(STATE_SIZE, dtype=np.float32)
            )

            buffer.push(obs, action, reward1, next_obs, done)

        episode += 1
        if episode % 100 == 0:
            print(f"  Predtrening: {episode}/{num_episodes} epizoda | buffer: {len(buffer)}")

    print(f"Predtrening zavrsen. Buffer: {len(buffer)} iskustava.")
    return buffer
