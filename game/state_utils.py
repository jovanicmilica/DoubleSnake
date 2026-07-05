import numpy as np
from collections import deque
from game.snake import Direction

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

MAX_SNAKE_LENGTH = 100  # BOARD_SIZE^2
MAX_DISTANCE = 20       # BOARD_SIZE * 2
STATE_SIZE = 22


def get_state(snake, other_snake, board):
    """
    Vraca feature vektor oblika (22,), dtype=float32:
      [0:4]   - opasnost 1 korak unapred (gore, dole, levo, desno)
      [4:8]   - trenutni pravac kretanja, one-hot (gore, dole, levo, desno)
      [8:12]  - relativni pravac hrane (gore, dole, levo, desno)
      [12:14] - normalizovane duzine (moja, protivnicka)
      [14]    - normalizovana udaljenost do protivnicke glave
      [15:19] - normalizovan dostupan prostor (flood fill) u svakom pravcu
      [19]    - moja normalizovana udaljenost do hrane
      [20]    - protivnicka normalizovana udaljenost do hrane
      [21]    - prednost u trci do hrane, 0..1; vise znaci da sam blizi
    """
    head = snake.head()
    blocked = set(snake.body[1:] + other_snake.body)
    opponent_next_positions = _possible_next_head_positions(other_snake, board, blocked)

    dangers = [
        _is_danger(head, d, board, blocked, opponent_next_positions)
        for d in DIRECTIONS
    ]
    current_dir = [1.0 if snake.direction == d else 0.0 for d in DIRECTIONS]
    food_dir = _food_direction(head, board)

    my_length = len(snake.body) / MAX_SNAKE_LENGTH
    enemy_length = len(other_snake.body) / MAX_SNAKE_LENGTH

    enemy_head = other_snake.head()
    dist_to_enemy = _manhattan(head, enemy_head) / MAX_DISTANCE

    space = _reachable_space_per_direction(head, board, blocked)
    my_food_dist, enemy_food_dist, food_race_advantage = _food_race_features(
        head,
        enemy_head,
        board,
    )

    state = (
        dangers
        + current_dir
        + food_dir
        + [my_length, enemy_length, dist_to_enemy]
        + space
        + [my_food_dist, enemy_food_dist, food_race_advantage]
    )
    return np.array(state, dtype=np.float32)


def _is_danger(head, direction, board, blocked, opponent_next_positions=None):
    """Da li bi pomeranje u datom pravcu iz trenutne pozicije odmah ubilo zmiju"""
    dr, dc = direction
    next_pos = (head[0] + dr, head[1] + dc)

    if board.is_out_of_bounds(next_pos):
        return 1.0
    if next_pos in blocked:
        return 1.0
    if opponent_next_positions is not None and next_pos in opponent_next_positions:
        return 1.0
    return 0.0


def _possible_next_head_positions(snake, board, blocked):
    reverse_direction = (-snake.direction[0], -snake.direction[1])
    positions = set()

    for direction in DIRECTIONS:
        if direction == reverse_direction:
            continue

        next_pos = (snake.head()[0] + direction[0], snake.head()[1] + direction[1])
        if board.is_out_of_bounds(next_pos) or next_pos in blocked:
            continue
        positions.add(next_pos)

    return positions


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


def _food_position(board):
    if board.food is None:
        return None
    return board.food[0] if isinstance(board.food, tuple) and len(board.food) >= 1 else board.food


def _food_race_features(my_head, enemy_head, board):
    food_pos = _food_position(board)
    if food_pos is None:
        return 1.0, 1.0, 0.5

    my_dist = _manhattan(my_head, food_pos)
    enemy_dist = _manhattan(enemy_head, food_pos)
    normalized_my_dist = min(my_dist / MAX_DISTANCE, 1.0)
    normalized_enemy_dist = min(enemy_dist / MAX_DISTANCE, 1.0)

    # 0.5 je izjednaceno; iznad 0.5 znaci da sam blizi hrani od protivnika.
    advantage = 0.5 + (enemy_dist - my_dist) / (2 * MAX_DISTANCE)
    advantage = max(0.0, min(1.0, advantage))

    return normalized_my_dist, normalized_enemy_dist, advantage


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
