from game.snake import Snake, Direction, make_snake1, make_snake2
from game.board import Board
from config.config import REWARD_FOOD


class GameState:
    ONGOING = "ongoing"
    SNAKE1_WIN = "snake1_win"
    SNAKE2_WIN = "snake2_win"
    DRAW = "draw"


class Game:
    def __init__(self):
        self.snake1 = make_snake1()
        self.snake2 = make_snake2()
        self.board = Board()
        self.state = GameState.ONGOING
        self.steps = 0

        # Postavljanje hrane na pocetku igre
        self.board.place_food(self.snake1.body, self.snake2.body)

    def step(self, dir1, dir2):
        """
        Vraca (reward1, reward2, done)
        """
        if self.state != GameState.ONGOING:
            return 0, 0, True

        self.steps += 1

        # postavljanje pravca kretanja zmija
        self.snake1.set_direction(dir1)
        self.snake2.set_direction(dir2)

        # pomjeranje zmija
        self.snake1.move()
        self.snake2.move()

        # provjera kolizije
        dead1 = self.board.check_collision(self.snake1, self.snake2)
        dead2 = self.board.check_collision(self.snake2, self.snake1)

        # ako su glave zmija na istoj poziciji, obe zmije umiru
        if self.snake1.head() == self.snake2.head():
            dead1 = True
            dead2 = True

        # postavljanje stanja zmija na osnovu kolizije
        if dead1:
            self.snake1.alive = False
        if dead2:
            self.snake2.alive = False

        # racunanje nagrada na osnovu stanja zmija
        reward1, reward2 = self._calculate_rewards(dead1, dead2)

        # provjera hrane i dodavanje nagrade ako je zmija pojela hranu
        if self.snake1.alive and self.board.check_food(self.snake1):
            reward1 += REWARD_FOOD
            self.snake1.grow()
            self.board.place_food(self.snake1.body, self.snake2.body)

        if self.snake2.alive and self.board.check_food(self.snake2):
            reward2 += REWARD_FOOD
            self.snake2.grow()
            self.board.place_food(self.snake1.body, self.snake2.body)

        done = self.state != GameState.ONGOING
        return reward1, reward2, done

    def _calculate_rewards(self, dead1, dead2):
        from config.config import REWARD_DEATH, REWARD_WIN, REWARD_STEP

        reward1 = REWARD_STEP
        reward2 = REWARD_STEP

        if dead1 and dead2:
            self.state = GameState.DRAW
            reward1 = REWARD_DEATH
            reward2 = REWARD_DEATH

        elif dead1:
            self.state = GameState.SNAKE2_WIN
            reward1 = REWARD_DEATH
            reward2 = REWARD_WIN

        elif dead2:
            self.state = GameState.SNAKE1_WIN
            reward1 = REWARD_WIN
            reward2 = REWARD_DEATH

        return reward1, reward2

    def get_state(self):
        return self.state

    def is_done(self):
        return self.state != GameState.ONGOING

    def reset(self):
        self.__init__()