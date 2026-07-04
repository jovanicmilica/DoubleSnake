import sys
import pygame

from agents.heuristic_agent.heuristic_agent import HeuristicAgent
from agents.rl_agent.policy import load_trained_agent
from agents.rl_agent.snake_env import ACTIONS
from game.game import Game, GameState
from game.renderer import Renderer
from game.state_utils import get_state


FPS = 8
MAX_GAMES = None


def run(render_live=True):
    agent, message = load_trained_agent()
    if agent is None:
        print(message)
        return

    renderer = Renderer("RL Agent", "Heuristic") if render_live else None
    scores = {"player1": 0, "player2": 0, "draws": 0}
    games_played = 0
    running = True

    while running:
        for event in pygame.event.get() if render_live else []:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        if MAX_GAMES is not None and games_played >= MAX_GAMES:
            running = False
        if not running:
            break

        game = Game()
        heuristic = HeuristicAgent(game.snake2, game.snake1, game.board)
        done = False

        while not done and running:
            if render_live:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        running = False
                if not running:
                    break

            obs = get_state(game.snake1, game.snake2, game.board)
            rl_dir = ACTIONS[agent.select_action(obs)]
            heuristic_dir = heuristic.get_action()

            _, _, done = game.step(rl_dir, heuristic_dir)
            heuristic = HeuristicAgent(game.snake2, game.snake1, game.board)

            if render_live:
                renderer.draw(game, scores, fps=FPS)

        if not running:
            break

        games_played += 1
        state = game.get_state()
        if state == GameState.SNAKE1_WIN:
            scores["player1"] += 1
        elif state == GameState.SNAKE2_WIN:
            scores["player2"] += 1
        else:
            scores["draws"] += 1

        if render_live:
            renderer.draw(game, scores, fps=FPS)
        elif games_played % 100 == 0:
            print(
                f"Games: {games_played} | "
                f"RL: {scores['player1']} | "
                f"Heuristic: {scores['player2']} | "
                f"Draws: {scores['draws']}"
            )

    if renderer is not None:
        renderer.quit()
    sys.exit()
