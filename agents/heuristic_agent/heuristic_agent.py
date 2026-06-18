import heapq
from collections import deque
from game.snake import Direction
from config.config import BOARD_SIZE

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class HeuristicAgent:
    def __init__(self, snake, other_snake, board):
        self.snake = snake
        self.other_snake = other_snake
        self.board = board

    def get_action(self):
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

        blocked = set(self.snake.body[1:] + self.other_snake.body)

        # heap element: (f, g, pozicija, put)
        heap = []
        heapq.heappush(heap, (0, 0, start, []))

        # najmanja g vrijednost do svakog posjećenog čvora
        visited = {start: 0}

        while heap:
            f, g, current, path = heapq.heappop(heap)

            if current == goal:
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
                h = self._manhattan(next_pos, goal)
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