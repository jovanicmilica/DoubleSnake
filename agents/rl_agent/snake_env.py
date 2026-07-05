import numpy as np
import gymnasium as gym

from game.game import Game, GameState
from game.snake import Direction
from game.state_utils import STATE_SIZE, get_state, _flood_fill
from agents.heuristic_agent.heuristic_agent import HeuristicAgent
from config.config import (
    MAX_STEPS_WITHOUT_FOOD,
    OPPONENT_RANDOM_PROB_END,
    OPPONENT_RANDOM_PROB_START,
    REWARD_CLOSER_TO_FOOD,
    REWARD_FARTHER_FROM_FOOD,
    REWARD_HIT_OPPONENT,
    REWARD_OPPONENT_FOOD,
    REWARD_PICKED_FOOD,
    REWARD_STALL,
)

ACTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class DoubleSnakeEnv(gym.Env):
    """
    Gymnasium wrapper oko Game klase.

    Agent (snake1) igra protiv HeuristicAgent-a (snake2).
    Observation: feature vektor iz state_utils.get_state()
    Action space: Discrete(4) — UP, DOWN, LEFT, RIGHT
    """

    metadata = {"render_modes": ["human"]} # igra se prikazuje na ekranu

    def __init__(self, render_mode=None):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0.0,    # low predtsavlja minimum vrednosti u feature vektoru 
            high=1.0,   # high predtsavlja maksimum vrednosti u feature vektoru
            shape=(STATE_SIZE,), # shape predtsavlja oblik feature vektora
            dtype=np.float32,   # dtype predtsavlja tip podataka feature vektora
        )

        self.action_space = gym.spaces.Discrete(4) # 4 moguce akcije 

        self.render_mode = render_mode
        self._renderer = None  # None jer se renderer kreira tek kad se pozove render()
        self.game = None # None jer se igra kreira tek kad se pozove reset()
        self.opponent = None # None jer se protivnik kreira tek kad se pozove reset()
        self.steps_without_food = 0
        self.agent_food_count = 0
        self.opponent_food_count = 0
        self.opponent_random_prob = OPPONENT_RANDOM_PROB_START

        # prati pobjede kroz sve epizode — renderer ocekuje ovaj format
        self.scores = {"player1": 0, "player2": 0}

    def reset(self, seed=None, options=None):
        '''
        reset() — obavezna Gymnasium metoda
        Kreira novu igru i vraca pocetno stanje (obs, info).
        obs - feature vektor float32
        info - dict sa dodatnim informacijama, za sada samo game_state
        '''
        
        super().reset(seed=seed) # za nasumicne dogadjaje

        self.game = Game()
        self.steps_without_food = 0
        self.agent_food_count = 0
        self.opponent_food_count = 0

        self.opponent = HeuristicAgent(
            self.game.snake2,
            self.game.snake1,
            self.game.board,
            random_action_prob=self.opponent_random_prob,
        )

        obs = self._get_obs()
        info = {}
        return obs, info


    def step(self, action):
        '''
        step — obavezna Gymnasium metoda
        action: int u [0, 3] koji predstavlja pravac kretanja snake1 (nas agent)
        Vracamo tuple (obs, reward, terminated, truncated, info)
        obs - feature vektor float32
        reward - float, nagrada za snake1 (nas agent)
        terminated - bool, True ako je igra zavrsena prirodno (zmija umrla, neko pobijedio)
        truncated - bool, True ako je igra prekinuta van razloga (npr. predugo traje)
        info - dict sa dodatnim informacijama
        '''
        assert self.game is not None, "Pozovi reset() prije step()-a"

        dir1 = ACTIONS[action]

        dir2 = self.opponent.get_action()

        previous_food = self._food_pos()
        previous_my_dist = self._distance_to_food(self.game.snake1.head(), previous_food)

        reward1, _, done = self.game.step(dir1, dir2) # nasa nagrada
        current_food = self._food_pos()

        if not self.game.snake1.alive and self.game.snake1.head() in self.game.snake2.body:
            reward1 += REWARD_HIT_OPPONENT

        if self.game.snake1.alive and previous_food is not None:
            if self.game.snake1.head() == previous_food:
                reward1 += REWARD_PICKED_FOOD
                self.steps_without_food = 0
                self.agent_food_count += 1
            elif self.game.snake2.alive and self.game.snake2.head() == previous_food:
                reward1 += REWARD_OPPONENT_FOOD
                self.steps_without_food = 0
                self.opponent_food_count += 1
            else:
                self.steps_without_food += 1
                current_my_dist = self._distance_to_food(self.game.snake1.head(), previous_food)
                if current_my_dist < previous_my_dist:
                    reward1 += REWARD_CLOSER_TO_FOOD
                elif current_my_dist > previous_my_dist:
                    reward1 += REWARD_FARTHER_FROM_FOOD

        if (
            not done
            and self.game.snake1.alive
            and self.game.snake2.alive
            and current_food is not None
            and self.steps_without_food >= MAX_STEPS_WITHOUT_FOOD
        ):
            reward1 += REWARD_STALL
            done = True

        # space control: mali bonus ako agent kontrolise vise prostora od protivnika
        # potice agenta da potiskuje heuristiku, a ne samo da prezivljava
        if self.game.snake1.alive and self.game.snake2.alive:
            # body[1:] iskljucuje glavu — flood fill mora startati iz glave koja nije blokirana
            s1_blocked = set(self.game.snake1.body[1:] + self.game.snake2.body)
            s2_blocked = set(self.game.snake2.body[1:] + self.game.snake1.body)
            my_space    = _flood_fill(self.game.snake1.head(), self.game.board, s1_blocked)
            their_space = _flood_fill(self.game.snake2.head(), self.game.board, s2_blocked)
            reward1 += 0.001 * (my_space - their_space)

        self.opponent = HeuristicAgent(
            self.game.snake2,
            self.game.snake1,
            self.game.board,
            random_action_prob=self.opponent_random_prob,
        )

        obs = self._get_obs()
        terminated = done
        truncated = False

        if terminated:
            state = self.game.get_state()
            if state == GameState.SNAKE1_WIN:
                self.scores["player1"] += 1
            elif state == GameState.SNAKE2_WIN:
                self.scores["player2"] += 1

        info = {
            "game_state": self.game.get_state(),
            "agent_food": self.agent_food_count,
            "opponent_food": self.opponent_food_count,
        }
        return obs, float(reward1), terminated, truncated, info


    def render(self):
        
        if self.render_mode != "human" or self.game is None:
            return

        if self._renderer is None:
            from game.renderer import Renderer
            self._renderer = Renderer("DQN Agent", "Heuristic")

        self._renderer.draw(self.game, self.scores, fps=10)

  
    def close(self):
        if self._renderer is not None:
            self._renderer.quit()
            self._renderer = None

    
    def _get_obs(self):
        '''
        Vraca feature vektor float32 koji predstavlja trenutno stanje igre.
        Ako je igra zavrsena (zmija mrtva), vraca vektor nula
        '''
        if not self.game.snake1.alive:
            return np.zeros(STATE_SIZE, dtype=np.float32)

        return get_state(self.game.snake1, self.game.snake2, self.game.board)

    def _food_pos(self):
        if self.game.board.food is None:
            return None
        return self.game.board.food[0]

    def _distance_to_food(self, head, food_pos):
        if food_pos is None:
            return 0
        return abs(head[0] - food_pos[0]) + abs(head[1] - food_pos[1])

    def set_opponent_random_prob(self, value):
        self.opponent_random_prob = max(OPPONENT_RANDOM_PROB_END, min(OPPONENT_RANDOM_PROB_START, value))
