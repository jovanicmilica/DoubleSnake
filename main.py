"""
Double Snake - glavna ulazna tacka.

Pokreni: python main.py
"""

import pygame

from game.start_screen import StartScreen


def main():
    choice = StartScreen().run()
    pygame.quit()

    if choice == 1:
        from game_modes.player_vs_rl_agent import run
        run()
    elif choice == 2:
        from game_modes.rl_vs_heuristic import run
        run(render_live=True)
    elif choice == 3:
        from agents.rl_agent.train import run
        run(render=False)


if __name__ == "__main__":
    main()
