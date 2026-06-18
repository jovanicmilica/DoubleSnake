"""
main_play.py – Player vs AI mode (Play mode)
  - Player controls snake1 with WASD or Arrow keys
  - HeuristicAgent controls snake2
  - Press R to restart, Q to quit
"""

import sys
import pygame

from game.game import Game
from game.game import GameState
from game.snake import Direction
from game.renderer import Renderer
from agents.heuristic_agent.heuristic_agent import HeuristicAgent


# constants
FPS = 5   # game ticks per second – raise for harder, lower for easier

KEY_DIR = {
    # WASD
    pygame.K_w: Direction.UP,
    pygame.K_s: Direction.DOWN,
    pygame.K_a: Direction.LEFT,
    pygame.K_d: Direction.RIGHT,
    # Arrow keys
    pygame.K_UP:    Direction.UP,
    pygame.K_DOWN:  Direction.DOWN,
    pygame.K_LEFT:  Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
}

RESULT_MSG = {
    GameState.SNAKE1_WIN: "YOU WIN!",
    GameState.SNAKE2_WIN: "AI WINS",
    GameState.DRAW:       "DRAW",
}


# helpers 

def make_agent(game):
    """Build a fresh HeuristicAgent bound to the current game objects."""
    return HeuristicAgent(game.snake2, game.snake1, game.board)


def run():
    renderer = Renderer("You", "AI")
    game = Game()
    agent = make_agent(game)
    scores = {"player1": 0, "player2": 0}
    player_dir = game.snake1.direction
    running = True
    game_over = False
    end_message = ""   # čuva poruku za kraj

    while running:
        # event handling 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif event.key == pygame.K_r:
                    # restart 
                    game = Game()
                    agent = make_agent(game)
                    player_dir = game.snake1.direction
                    game_over = False
                    end_message = ""
                    continue   

                elif event.key in KEY_DIR and not game_over:
                    player_dir = KEY_DIR[event.key]

        if not running:
            break

        # game tick 
        if not game_over:
            ai_dir = agent.get_action()
            _, _, done = game.step(player_dir, ai_dir)
            agent = make_agent(game)

            if done:
                game_over = True
                state = game.get_state()
                if state == GameState.SNAKE1_WIN:
                    scores["player1"] += 1
                elif state == GameState.SNAKE2_WIN:
                    scores["player2"] += 1
                end_message = RESULT_MSG.get(state, "GAME OVER")

                # End 
                renderer.draw_end_screen(end_message, scores)

        # rendering 
        if not game_over:
            # Draw board untill it's over
            renderer.draw(game, scores, fps=FPS)
        else:
            # When game over stop drawing
            renderer.tick(FPS)

    renderer.quit()
    sys.exit()


if __name__ == "__main__":
    run()