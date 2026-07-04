import random
import os
from config.config import BOARD_SIZE


class Board:
    def __init__(self):
        self.size = BOARD_SIZE
        self.food = None
        # dostupni tipovi voca
        assets_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'assets', 'fruits')
        )
        if os.path.isdir(assets_dir):
            types = [os.path.splitext(f)[0] for f in os.listdir(assets_dir) if f.lower().endswith('.png')]
            if types:
                self._fruit_types = types
            else:
                self._fruit_types = ['apple', 'pear', 'blueberry', 'orange', 'grapes']
        else:
            self._fruit_types = ['apple', 'pear', 'blueberry', 'orange', 'grapes']

    def place_food(self, snake1_body, snake2_body):
        # celije koje su zauzete od strane zmija
        occupied = set(snake1_body + snake2_body)
        
        # slobodne celije
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in occupied
        ]

        if free:
            pos = random.choice(free)
            fruit = random.choice(self._fruit_types)
            self.food = (pos, fruit)
        else:
            self.food = None  

    def is_out_of_bounds(self, position):
        r, c = position
        return r < 0 or r >= self.size or c < 0 or c >= self.size

    def check_collision(self, snake, other_snake):
        '''
        Vraca True ako je zmija udarila u zid, samu sebe ili drugu zmiju, inace False.
        '''
        head = snake.head()

        if self.is_out_of_bounds(head):
            return True

        if head in snake.body[1:]:
            return True

        if head in other_snake.body:
            return True

        return False

    def check_food(self, snake):
        if self.food is None:
            return False
        pos = self.food[0] if isinstance(self.food, tuple) and len(self.food) >= 1 else self.food
        return snake.head() == pos