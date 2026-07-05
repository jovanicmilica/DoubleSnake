import sys
import pygame

from agents.rl_agent.policy import load_trained_agent
from agents.rl_agent.snake_env import ACTIONS
from game.game import Game, GameState
from game.renderer import Renderer, TEXT_DIM, TEXT_MAIN, board_pixel_size
from game.snake import Direction
from game.state_utils import get_state


FPS = 5

KEY_DIR = {
    pygame.K_w: Direction.UP,
    pygame.K_s: Direction.DOWN,
    pygame.K_a: Direction.LEFT,
    pygame.K_d: Direction.RIGHT,
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
}

RESULT_MSG = {
    GameState.SNAKE1_WIN: "YOU WIN!",
    GameState.SNAKE2_WIN: "RL WINS",
    GameState.DRAW: "DRAW",
}


def run():
    agent, message = load_trained_agent()
    if agent is None:
        _show_message(message)
        return

    renderer = Renderer("You", "RL Agent")
    game = Game()
    scores = {"player1": 0, "player2": 0}
    player_dir = game.snake1.direction
    running = True
    game_over = False
    end_message = ""

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    game = Game()
                    player_dir = game.snake1.direction
                    game_over = False
                    end_message = ""
                elif event.key in KEY_DIR and not game_over:
                    player_dir = KEY_DIR[event.key]

        if not running:
            break

        if not game_over:
            rl_obs = get_state(game.snake2, game.snake1, game.board)
            rl_action = agent.select_action(rl_obs)
            rl_dir = ACTIONS[rl_action]

            _, _, done = game.step(player_dir, rl_dir)
            if done:
                game_over = True
                state = game.get_state()
                if state == GameState.SNAKE1_WIN:
                    scores["player1"] += 1
                elif state == GameState.SNAKE2_WIN:
                    scores["player2"] += 1
                end_message = RESULT_MSG.get(state, "GAME OVER")
                renderer.draw_end_screen(end_message, scores)

        if not game_over:
            renderer.draw(game, scores, fps=FPS)
        else:
            renderer.tick(FPS)

    renderer.quit()
    sys.exit()


def _show_message(message):
    pygame.init()
    board_px = board_pixel_size()
    screen = pygame.display.set_mode((board_px + 240, board_px))
    pygame.display.set_caption("Double Snake")
    font_lg = pygame.font.SysFont("consolas", 24, bold=True)
    font_sm = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE, pygame.K_RETURN):
                running = False

        screen.fill((15, 15, 20))
        title = font_lg.render("RL agent nije spreman", True, TEXT_MAIN)
        detail = font_sm.render(message, True, TEXT_DIM)
        hint = font_sm.render("Q / Enter za nazad", True, TEXT_DIM)
        screen.blit(title, title.get_rect(center=(screen.get_width() // 2, 210)))
        screen.blit(detail, detail.get_rect(center=(screen.get_width() // 2, 250)))
        screen.blit(hint, hint.get_rect(center=(screen.get_width() // 2, 300)))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
