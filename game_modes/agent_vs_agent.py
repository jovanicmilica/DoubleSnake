"""
  - HeuristicAgent vs HeuristicAgent
  - Continuously plays games between two agents, tracking win rates and draws
  - Press Q to quit
"""

import sys
import pygame

from game.game import Game
from game.game import GameState
from game.renderer import Renderer
from agents.heuristic_agent.heuristic_agent import HeuristicAgent


FPS = 5   # game ticks per second – raise for harder, lower for easier

MAX_GAMES = 4  # None means no limit, keep playing until user quits

RENDER_LIVE = True  # render the game in real-time, or just run the games in the background and show final results

RESULT_MSG = {
    GameState.SNAKE1_WIN: "AGENT 1 WINS",
    GameState.SNAKE2_WIN: "AGENT 2 WINS",
    GameState.DRAW:       "DRAW",
}

def make_agents(game):
    """
    Builds two HeuristicAgents bound to the current game objects.
    """
    agent1 = HeuristicAgent(game.snake1, game.snake2, game.board)
    agent2 = HeuristicAgent(game.snake2, game.snake1, game.board)
    return agent1, agent2


def run():
    """
    Runs a continuous loop of games between two HeuristicAgents, tracking win rates and draws.
    """
    renderer = Renderer("Agent1", "Agent2")
    
    scores = {"player1": 0, "player2": 0, "draws": 0}
    games_played = 0
    
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:   # User closed the window
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        if MAX_GAMES is not None and games_played >= MAX_GAMES:
            running = False
        
        if not running:
            break
        
        # start a new game
        game = Game()
        agent1, agent2 = make_agents(game)
        game_over = False
        
        while not game_over:
            # each agent decides on its next move
            agent1_dir = agent1.get_action()
            agent2_dir = agent2.get_action()
            
            # both agents make their moves simultaneously
            _, _, done = game.step(agent1_dir, agent2_dir)
            
            agent1, agent2 = make_agents(game)
            
            # render the whole game in real-time
            if RENDER_LIVE:
                renderer.draw(game, {"player1": scores["player1"], "player2": scores["player2"]}, fps=FPS)
            
            if done:
                game_over = True
        
        # update scores 
        games_played += 1
        state = game.get_state()
        if state == GameState.SNAKE1_WIN:
            scores["player1"] += 1
        elif state == GameState.SNAKE2_WIN:
            scores["player2"] += 1
        elif state == GameState.DRAW:
            scores["draws"] += 1
        
        # render the final state of the game
        renderer.draw(game, scores, fps=FPS)
        
        # console output 
        if games_played % 100 == 0:
            print(f"Games played: {games_played}")
            print(f"   Agent1: {scores['player1']} wins")
            print(f"   Agent2: {scores['player2']} wins")
            print(f"   Draws: {scores['draws']}")
            print(f"   Win rate Agent1: {scores['player1']/games_played*100:.1f}%")
            print("---")
    
    print(f"\n Training completed!")
    print(f"Total games played: {games_played}")
    print(f"Agent1 wins: {scores['player1']} ({scores['player1']/games_played*100:.1f}%)")
    print(f"Agent2 wins: {scores['player2']} ({scores['player2']/games_played*100:.1f}%)")
    print(f"Draws: {scores['draws']} ({scores['draws']/games_played*100:.1f}%)")
    
    renderer.quit()
    sys.exit()


if __name__ == "__main__":
    run()