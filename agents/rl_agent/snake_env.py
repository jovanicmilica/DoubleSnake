import numpy as np
import gymnasium as gym

from game.game import Game, GameState
from game.snake import Direction
from game.state_utils import get_state, _flood_fill
from agents.heuristic_agent.heuristic_agent import HeuristicAgent

ACTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class DoubleSnakeEnv(gym.Env):
    """
    Gymnasium wrapper oko Game klase.

    Agent (snake1) igra protiv HeuristicAgent-a (snake2).
    Observation: feature vektor (19,) iz state_utils.get_state()
    Action space: Discrete(4) — UP, DOWN, LEFT, RIGHT
    """

    metadata = {"render_modes": ["human"]} # igra se prikazuje na ekranu

    def __init__(self, render_mode=None):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0.0,    # low predtsavlja minimum vrednosti u feature vektoru 
            high=1.0,   # high predtsavlja maksimum vrednosti u feature vektoru
            shape=(19,), # shape predtsavlja oblik feature vektora
            dtype=np.float32,   # dtype predtsavlja tip podataka feature vektora
        )

        self.action_space = gym.spaces.Discrete(4) # 4 moguce akcije 

        self.render_mode = render_mode
        self._renderer = None  # None jer se renderer kreira tek kad se pozove render()
        self.game = None # None jer se igra kreira tek kad se pozove reset()
        self.opponent = None # None jer se protivnik kreira tek kad se pozove reset()

        # prati pobjede kroz sve epizode — renderer ocekuje ovaj format
        self.scores = {"player1": 0, "player2": 0}

    def reset(self, seed=None, options=None):
        '''
        reset() — obavezna Gymnasium metoda
        Kreira novu igru i vraca pocetno stanje (obs, info).
        obs - feature vektor (19,) float32
        info - dict sa dodatnim informacijama, za sada samo game_state
        '''
        
        super().reset(seed=seed) # za nasumicne dogadjaje

        self.game = Game()

        self.opponent = HeuristicAgent(
            self.game.snake2,
            self.game.snake1,
            self.game.board,
        )

        obs = self._get_obs()
        info = {}
        return obs, info


    def step(self, action):
        '''
        step — obavezna Gymnasium metoda
        action: int u [0, 3] koji predstavlja pravac kretanja snake1 (nas agent)
        Vracamo tuple (obs, reward, terminated, truncated, info)
        obs - feature vektor (19,) float32
        reward - float, nagrada za snake1 (nas agent)
        terminated - bool, True ako je igra zavrsena prirodno (zmija umrla, neko pobijedio)
        truncated - bool, True ako je igra prekinuta van razloga (npr. predugo traje)
        info - dict sa dodatnim informacijama
        '''
        assert self.game is not None, "Pozovi reset() prije step()-a"

        dir1 = ACTIONS[action]

        dir2 = self.opponent.get_action()

        reward1, _, done = self.game.step(dir1, dir2) # nasa nagrada

        # space control: mali bonus ako agent kontrolise vise prostora od protivnika
        # potice agenta da potiskuje heuristiku, a ne samo da prezivljava
        if self.game.snake1.alive and self.game.snake2.alive:
            # body[1:] iskljucuje glavu — flood fill mora startati iz glave koja nije blokirana
            s1_blocked = set(self.game.snake1.body[1:] + self.game.snake2.body)
            s2_blocked = set(self.game.snake2.body[1:] + self.game.snake1.body)
            my_space    = _flood_fill(self.game.snake1.head(), self.game.board, s1_blocked)
            their_space = _flood_fill(self.game.snake2.head(), self.game.board, s2_blocked)
            reward1 += 0.005 * (my_space - their_space)

        self.opponent = HeuristicAgent(
            self.game.snake2,
            self.game.snake1,
            self.game.board,
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

        info = {"game_state": self.game.get_state()}
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
        Vraca feature vektor (19,) float32 koji predstavlja trenutno stanje igre.
        Ako je igra zavrsena (zmija mrtva), vraca vektor nula
        '''
        if not self.game.snake1.alive:
            return np.zeros(19, dtype=np.float32)

        return get_state(self.game.snake1, self.game.snake2, self.game.board)