import random
from config import BOARD_SIZE


class Board:
    def __init__(self):
        self.size = BOARD_SIZE
        self.food = None

    def place_food(self, snake1_body, snake2_body):
        # occupied cells by both snakes
        occupied = set(snake1_body + snake2_body)
        
        # free cells are those not occupied by either snake
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in occupied
        ]

        if free:
            self.food = random.choice(free)
        else:
            self.food = None  

    def is_out_of_bounds(self, position):
        r, c = position
        return r < 0 or r >= self.size or c < 0 or c >= self.size

    def check_collision(self, snake, other_snake):
        '''
        Returns True if the snake has collided with a wall, itself, or the other snake.
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
        return snake.head() == self.food