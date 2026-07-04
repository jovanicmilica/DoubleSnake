import heapq
import random
from collections import deque
from game.snake import Direction
from config.config import BOARD_SIZE, HEURISTIC_ACTION_RANDOM_PROB

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class HeuristicAgent:
    def __init__(self, snake, other_snake, board, random_action_prob=HEURISTIC_ACTION_RANDOM_PROB):
        self.snake = snake
        self.other_snake = other_snake
        self.board = board
        # sansa da se, umjesto A* poteza, odigra potpuno random potez.
        # Uvodi raznovrsnost u iskustva koja zavrse u heuristic bufferu
        # (deterministicki agent bi uvijek posjecivao iste, uske putanje).
        self.random_action_prob = random_action_prob

    def get_action(self):
        if self.random_action_prob > 0 and random.random() < self.random_action_prob:
            return random.choice(DIRECTIONS)

        path = self._astar()
        if path:
            return path[0]
        return self._safe_direction()

    def _manhattan(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _astar(self):
        start = self.snake.head()

        goal = self.board.food

        if goal is None:
            return None

        if isinstance(goal, tuple) and len(goal) >= 1 and isinstance(goal[0], tuple):
            goal_pos = goal[0]
        else:
            goal_pos = goal

        blocked = set(self.snake.body[1:] + self.other_snake.body)

        heap = []
        heapq.heappush(heap, (0, 0, start, []))

        visited = {start: 0}

        while heap:
            f, g, current, path = heapq.heappop(heap)

            if current == goal_pos:
                return path

            for direction in DIRECTIONS:
                dr, dc = direction
                r, c = current
                next_pos = (r + dr, c + dc)
                next_g = g + 1

                if (self.board.is_out_of_bounds(next_pos) or
                        next_pos in blocked):
                    continue

                # posjecujemo samo ako smo nasli bolji put
                if next_pos in visited and visited[next_pos] <= next_g:
                    continue

                visited[next_pos] = next_g
                h = self._manhattan(next_pos, goal_pos)
                f_new = next_g + h
                heapq.heappush(heap, (f_new, next_g, next_pos, path + [direction]))

        return None

    def _safe_direction(self):
        head = self.snake.head()
        blocked = set(self.snake.body[1:] + self.other_snake.body)

        for direction in DIRECTIONS:
            dr, dc = direction
            r, c = head
            next_pos = (r + dr, c + dc)

            if (not self.board.is_out_of_bounds(next_pos) and
                    next_pos not in blocked):
                return direction

        return self.snake.direction