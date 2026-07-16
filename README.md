# Double Snake — RL Agent

A competitive two-player variant of the classic Snake game, where a
**Deep Q-Network (DQN)** agent is trained via reinforcement learning to
compete against a heuristic (A*) opponent.

## Overview

Two snakes play simultaneously on a 10x10 board. Each snake collects food
to grow and tries to outlive its opponent. Colliding with a wall, its own
body, or the opponent ends the game for that snake; a head-on collision
between both snakes results in a draw. The last snake standing wins.

## Features

- **Game engine** — snake movement, collision detection, food spawning,
  and win/draw conditions, fully independent of any AI logic.
- **Heuristic opponent** — an A* agent (Manhattan heuristic) that
  pathfinds to food, with a safe-direction fallback and configurable
  random-move probability for exploration/curriculum purposes.
- **State representation** — a compact 22-value normalized feature
  vector (danger flags, current direction, food direction, snake
  lengths, distance to opponent, flood-fill reachable space, and food-race
  metrics) instead of a raw image, enabling fast training with a small
  network.
- **DQN agent** — a fully-connected network (22 → 256 → 256 → 128 → 4)
  trained with:
  - Experience replay buffer (50k capacity, batch size 64)
  - Epsilon-greedy exploration (1.0 → 0.05)
  - Double DQN target computation (reduces Q-value overestimation)
  - A separate, periodically-synced target network for stable targets
  - Action masking (never selects an immediately fatal move)
  - Predictive danger (accounts for the opponent's possible next moves)
  - Space-control reward shaping and gradient clipping
- **Pretraining** — the replay buffer is seeded with heuristic-vs-heuristic
  demonstrations, with the sampling ratio linearly decaying to 0 as
  self-play experience accumulates.

## Game Modes

- **Train DQN Agent** — self-play training against the heuristic
  opponent, with optional live pygame rendering.
- **Agent vs Agent** — evaluates a trained DQN agent against the
  heuristic (A*) agent, no further learning occurs.
- **Player vs Agent** — a human plays against the trained agent in real
  time via keyboard controls (WASD / arrow keys).

## Project Structure

```
DoubleSnake/
├── main.py             # entry point — mode selection menu
├── config/              # all hyperparameters and constants
├── game/                 # game engine, state features, rendering
├── agents/
│   ├── heuristic_agent/  # A* opponent
│   └── rl_agent/         # DQN network, agent, replay buffer,
│                         #   Gym environment, training, pretraining
├── game_modes/           # playable mode scripts
└── checkpoints/          # saved model checkpoints (.pt)
```

## Requirements

- Python 3.x
- pygame
- torch
- gymnasium
- numpy

## Running

```bash
python main.py
```

Use the menu to select a mode (train, agent vs agent, or player vs
agent). Training checkpoints are saved periodically to `checkpoints/`
and can be resumed automatically from `latest.pt`.

## Authors

- Danica Komatović — SV 20/2023
- Milica Jovanić — SV 9/2023
