# Dimenzije table (10x10)
BOARD_SIZE = 10

# Startne pozicije zmija
SNAKE1_START = [(0, 1), (0, 0)]          
SNAKE2_START = [(9, 8), (9, 9)]          

# Pocetni pravci zmija (gore, dole, levo, desno)
SNAKE1_START_DIR = (0, 1)               # ide desno
SNAKE2_START_DIR = (0, -1)              # ide lijevo

# Nagrade 
REWARD_FOOD = 10
REWARD_DEATH = -100
REWARD_WIN = 200
REWARD_STEP = 0.1                       # mali bonus za prezivljavanje 

# Trening
REPLAY_BUFFER_SIZE = 50_000
HEURISTIC_BUFFER_SIZE = 10_000

BATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99999
LEARNING_RATE = 0.0001

GAMMA = 0.99                            # faktor umanjivanja buducih nagrada (discount factor)

TARGET_UPDATE_FREQ = 1000
PRETRAIN_EPISODES = 50000
HEURISTIC_PRETRAIN_EPISODES = 2_000    # heuristic vs heuristic epizode za punjenje buffera
HEURISTIC_ACTION_RANDOM_PROB = 0.08    # sansa da heuristicki agent napravi random potez umjesto A* poteza

# Mesanje heuristic i self-play buffera na pocetku DQN treninga.
# Na pocetku treninga sansa je HEURISTIC_SAMPLE_PROB_START da se batch za train_step
# uzme iz heuristic buffera, a zatim linearno opada do 0 kroz HEURISTIC_SAMPLE_DECAY_STEPS
# koraka treninga (nakon toga se trenira iskljucivo iz sopstvenog (self-play) buffera).
HEURISTIC_SAMPLE_PROB_START = 0.9
HEURISTIC_SAMPLE_DECAY_STEPS = 150_000

# Pygame vizuelizacija
CELL_SIZE = 60                         
FPS = 10