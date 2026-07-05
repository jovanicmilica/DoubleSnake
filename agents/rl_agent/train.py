"""
Petlja treniranja DQN agenta.
Pokreni: python -m agents.rl_agent.train
"""

import os
import sys
import pygame

from agents.rl_agent.snake_env import DoubleSnakeEnv
from agents.rl_agent.dqn_agent import DQNAgent
from config.config import (
    EPSILON_END,
    OPPONENT_RANDOM_DECAY_EPISODES,
    OPPONENT_RANDOM_PROB_END,
    OPPONENT_RANDOM_PROB_START,
    PRETRAIN_EPISODES,
)

CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_FREQ = 500   # sacuvaj svakih N epizoda
LOG_FREQ        = 100   # ispisi statistiku svakih N epizoda
RENDER          = False # True = gledas igru uzivo, False = samo trening


def run(render=None):
    render_enabled = RENDER if render is None else render
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if render_enabled:
        pygame.init()

    env   = DoubleSnakeEnv(render_mode="human" if render_enabled else None)
    agent = DQNAgent()

    start_episode = 0
    scores = {"player1": 0, "player2": 0, "draws": 0}

    # nastavi trening ako postoji latest checkpoint, inace predtreniranje heuristikom
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.exists(latest):
        try:
            start_episode, scores = agent.load(latest)
            env.scores["player1"] = scores["player1"]
            env.scores["player2"] = scores["player2"]
            # ako je epsilon vec pri minimumu, resetuj ga da agent nastavlja istrazivanje
            if agent.epsilon <= EPSILON_END * 1.5:
                agent.epsilon = 0.25
                print(f"Nastavljam od epizode {start_episode}, epsilon resetovan na {agent.epsilon:.3f}")
            else:
                print(f"Nastavljam od epizode {start_episode}, epsilon={agent.epsilon:.3f}")
        except RuntimeError as exc:
            print("Greska pri ucitavanju checkpointa; krecem novi trening.")
            print(f"Detalj: {exc}")
            from agents.rl_agent.pretrain import pretrain_buffer
            agent.set_heuristic_buffer(pretrain_buffer())
    else:
        print("Predtreniranje heuristic buffera heuristic agentom...")
        from agents.rl_agent.pretrain import pretrain_buffer
        agent.set_heuristic_buffer(pretrain_buffer())
        print(
            f"Heuristic buffer napunjen ({len(agent.heuristic_buffer)} iskustava). "
            f"Pocinjem DQN trening sa epsilon={agent.epsilon:.3f}, "
            f"heuristic_sample_prob={agent.heuristic_sample_prob:.3f}"
        )

    recent_rewards = []  # za pracenje prosjecne nagrade zadnjih 100 epizoda
    recent_agent_food = []
    recent_opponent_food = []

    for episode in range(start_episode, PRETRAIN_EPISODES):
        opponent_random_prob = _opponent_random_prob(episode)
        env.set_opponent_random_prob(opponent_random_prob)

        # provjeri da li korisnik zatvorio prozor (samo kad je render aktivan)
        if render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _finish(agent, episode, scores, env)
                    return

        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        # jedna epizoda igre
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, done)
            agent.train_step()

            obs = next_obs
            total_reward += reward

            if render_enabled:
                env.render()

        # azuriraj scores
        from game.game import GameState
        state = info["game_state"]
        if state == GameState.SNAKE1_WIN:
            scores["player1"] += 1
        elif state == GameState.SNAKE2_WIN:
            scores["player2"] += 1
        else:
            scores["draws"] += 1

        recent_rewards.append(total_reward)
        recent_agent_food.append(info.get("agent_food", 0))
        recent_opponent_food.append(info.get("opponent_food", 0))
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_agent_food.pop(0)
            recent_opponent_food.pop(0)

        # logovanje
        if (episode + 1) % LOG_FREQ == 0:
            games = episode + 1
            avg   = sum(recent_rewards) / len(recent_rewards)
            avg_food = sum(recent_agent_food) / len(recent_agent_food)
            avg_opp_food = sum(recent_opponent_food) / len(recent_opponent_food)
            wr    = scores["player1"] / games * 100
            log_line = (
                f"Ep {games:5d} | "
                f"avg reward (100): {avg:7.1f} | "
                f"food: {avg_food:.2f}/{avg_opp_food:.2f} | "
                f"epsilon: {agent.epsilon:.3f} | "
                f"opp_rand: {opponent_random_prob:.3f} | "
                f"win rate: {wr:.1f}% | "
                f"buffer: {len(agent.buffer)} | "
                f"heuristic_p: {agent.heuristic_sample_prob:.3f}"
            )
            print(log_line)
            with open("training_log.txt", "a", encoding="utf-8") as f:
                f.write(log_line + "\n")


        # checkpoint
        if (episode + 1) % CHECKPOINT_FREQ == 0:
            path = os.path.join(CHECKPOINT_DIR, f"ep_{episode+1}.pt")
            agent.save(path, episode + 1, scores)
            agent.save(latest, episode + 1, scores)
            print(f"  -> checkpoint sacuvan: {path}")

    _finish(agent, PRETRAIN_EPISODES, scores, env)


def _finish(agent, episode, scores, env):
    agent.save(os.path.join(CHECKPOINT_DIR, "latest.pt"), episode, scores)
    env.close()
    games = max(episode, 1)
    print("\nTrening zavrsen.")
    print(f"Odigrano epizoda : {games}")
    print(f"Agent pobjede    : {scores['player1']} ({scores['player1']/games*100:.1f}%)")
    print(f"Heuristic pobjede: {scores['player2']} ({scores['player2']/games*100:.1f}%)")
    print(f"Nerijeseno       : {scores['draws']}   ({scores['draws']/games*100:.1f}%)")
    sys.exit()


def _opponent_random_prob(episode):
    progress = min(episode / OPPONENT_RANDOM_DECAY_EPISODES, 1.0)
    return OPPONENT_RANDOM_PROB_START + progress * (
        OPPONENT_RANDOM_PROB_END - OPPONENT_RANDOM_PROB_START
    )
