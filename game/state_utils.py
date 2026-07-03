"""
izvlaenje feature vektora (state representation) za DQN agenta.


"""

import numpy as np
from collections import deque
from game.snake import Direction

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

MAX_SNAKE_LENGTH = 100  # BOARD_SIZE^2
MAX_DISTANCE = 20       # BOARD_SIZE * 2


def get_state(snake, other_snake, board):
    """
    Vraca feature vektor oblika (19,), dtype=float32:
      [0:4]   - opasnost 1 korak unapred (gore, dole, levo, desno)
      [4:8]   - trenutni pravac kretanja, one-hot (gore, dole, levo, desno)
      [8:12]  - relativni pravac hrane (gore, dole, levo, desno)
      [12:14] - normalizovane duzine (moja, protivnicka)
      [14]    - normalizovana udaljenost do protivnicke glave
      [15:19] - normalizovan dostupan prostor (flood fill) u svakom pravcu
    """
    head = snake.head()
    blocked = set(snake.body[1:] + other_snake.body)

    dangers = [_is_danger(head, d, board, blocked) for d in DIRECTIONS]
    current_dir = [1.0 if snake.direction == d else 0.0 for d in DIRECTIONS]
    food_dir = _food_direction(head, board)

    my_length = len(snake.body) / MAX_SNAKE_LENGTH
    enemy_length = len(other_snake.body) / MAX_SNAKE_LENGTH

    enemy_head = other_snake.head()
    dist_to_enemy = _manhattan(head, enemy_head) / MAX_DISTANCE

    space = _reachable_space_per_direction(head, board, blocked)

    state = dangers + current_dir + food_dir + [my_length, enemy_length, dist_to_enemy] + space
    return np.array(state, dtype=np.float32)


def _is_danger(head, direction, board, blocked):
    """Da li bi pomeranje u datom pravcu iz trenutne pozicije odmah ubilo zmiju"""
    dr, dc = direction
    next_pos = (head[0] + dr, head[1] + dc)

    if board.is_out_of_bounds(next_pos):
        return 1.0
    if next_pos in blocked:
        return 1.0
    return 0.0


def _food_direction(head, board):
    """Relativni pravac hrane: [gore, dole, levo, desno]. Mogu biti 2 aktivna istovremeno."""
    if board.food is None:
        return [0.0, 0.0, 0.0, 0.0]

    food_pos = board.food[0]
    fr, fc = food_pos
    hr, hc = head

    return [
        1.0 if fr < hr else 0.0,
        1.0 if fr > hr else 0.0,
        1.0 if fc < hc else 0.0,
        1.0 if fc > hc else 0.0,
    ]


def _manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def _flood_fill(start, board, blocked):
    """
    BFS flood fill iz `start` pozicije. Broji dostupne ćelije tretirajuci
    tela obe zmije kao prepreke.

    Vraca broj dostupnih celija.
    """
    if board.is_out_of_bounds(start) or start in blocked:
        return 0

    visited = {start}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS:
            npos = (r + dr, c + dc)
            if npos in visited or board.is_out_of_bounds(npos) or npos in blocked:
                continue
            visited.add(npos)
            queue.append(npos)

    return len(visited)


def _reachable_space_per_direction(head, board, blocked):
    """
    Za svaki od 4 pravca, izracunava koliko je celija dostupno (flood fill)
    ako bi zmija krenula u tom pravcu.
    """
    space = []
    for d in DIRECTIONS:
        next_pos = (head[0] + d[0], head[1] + d[1])
        reachable = _flood_fill(next_pos, board, blocked)
        space.append(reachable / (board.size * board.size))
    return space