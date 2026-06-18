# Board dimensions
BOARD_SIZE = 10

# Start positions of the snakes
SNAKE1_START = [(0, 1), (0, 0)]          
SNAKE2_START = [(9, 8), (9, 9)]          

# Start directions
SNAKE1_START_DIR = (0, 1)               # goes right 
SNAKE2_START_DIR = (0, -1)              # goes left

# Awards 
REWARD_FOOD = 10
REWARD_DEATH = -100
REWARD_WIN = 200
REWARD_STEP = -0.1                      # to avoid circling

# Training
REPLAY_BUFFER_SIZE = 50_000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
GAMMA = 0.99                            # discount factor
TARGET_UPDATE_FREQ = 1000              
PRETRAIN_EPISODES = 2000               

# Pygame visualizations
CELL_SIZE = 60                         
FPS = 10