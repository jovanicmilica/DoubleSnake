# Dimenzije table (10x10)
BOARD_SIZE = 10

# Startne pozicije zmija
SNAKE1_START = [(0, 1), (0, 0)]          
SNAKE2_START = [(9, 8), (9, 9)]          

# Pocetni pravci zmija (gore, dole, levo, desno)
SNAKE1_START_DIR = (0, 1)               # ide desno
SNAKE2_START_DIR = (0, -1)              # ide lijevo

# Nagrade 
REWARD_FOOD = 25
REWARD_DEATH = -100
REWARD_WIN = 200
REWARD_STEP = -0.02                     # mala kazna po koraku da agent ne kruzi beskonacno
REWARD_CLOSER_TO_FOOD = 0.2             # bonus kad se agent priblizi hrani
REWARD_FARTHER_FROM_FOOD = -0.2         # kazna kad se agent udalji od hrane
REWARD_PICKED_FOOD = 15                 # dodatni signal RL env-u za ulazak tacno u hranu
REWARD_OPPONENT_FOOD = -15              # kazna kad protivnik pojede hranu
REWARD_HIT_OPPONENT = -40               # dodatna kazna za sudar sa protivnikom
REWARD_STALL = -25                      # kazna za predugo kruzenje bez hrane
MAX_STEPS_WITHOUT_FOOD = 40             # prekida epizodu ako dugo niko ne jede

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
OPPONENT_RANDOM_PROB_START = 0.35       # slabiji protivnik na pocetku RL treninga
OPPONENT_RANDOM_PROB_END = 0.08         # finalna jacina protivnika
OPPONENT_RANDOM_DECAY_EPISODES = 12_000 # koliko epizoda traje postepeno pojacavanje protivnika

# Mesanje heuristic i self-play buffera na pocetku DQN treninga.
# Na pocetku treninga sansa je HEURISTIC_SAMPLE_PROB_START da se batch za train_step
# uzme iz heuristic buffera, a zatim linearno opada do 0 kroz HEURISTIC_SAMPLE_DECAY_STEPS
# koraka treninga (nakon toga se trenira iskljucivo iz sopstvenog (self-play) buffera).
HEURISTIC_SAMPLE_PROB_START = 0.9
HEURISTIC_SAMPLE_DECAY_STEPS = 150_000

# Pygame vizuelizacija
CELL_SIZE = 60                         
FPS = 10
