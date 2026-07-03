import numpy as np
from game.game import Game
from game.snake import Snake, Direction
from game.state_utils import get_state, _is_danger, _food_direction, _manhattan, _flood_fill, _reachable_space_per_direction, DIRECTIONS

# --- pomocna funkcija ---
def check(name, result, expected):
    ok = np.allclose(result, expected) if isinstance(expected, (list, np.ndarray)) else result == expected
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        print(f"         dobijeno:  {result}")
        print(f"         ocekivano: {expected}")

print("=" * 50)
print("TEST: state_utils.get_state")
print("=" * 50)

# Test 1: oblik i tip izlaznog vektora
print("\n--- Test 1: oblik i dtype ---")
game = Game()
state = get_state(game.snake1, game.snake2, game.board)
check("shape je (19,)", state.shape, (19,))
check("dtype je float32", state.dtype, np.float32)
check("sve vrednosti su u [0, 1]", bool(np.all(state >= 0) and np.all(state <= 1)), True)

# Test 2: detekcija opasnosti
# Snake1 pocetna pozicija: glava (0,1), telo [(0,1),(0,0)], pravac RIGHT
# Opasnost UP: (-1,1) -> van table = 1.0
# Opasnost DOWN: (1,1) -> slobodno = 0.0
# Opasnost LEFT: (0,0) -> telo zmije = 1.0
# Opasnost RIGHT: (0,2) -> slobodno = 0.0
print("\n--- Test 2: detekcija opasnosti (bits [0:4]) ---")
danger_bits = state[0:4]
check("gore = opasnost (van table)", danger_bits[0], 1.0)   # UP
check("dole = slobodno",             danger_bits[1], 0.0)   # DOWN
check("levo = opasnost (telo)",      danger_bits[2], 1.0)   # LEFT
check("desno = slobodno",            danger_bits[3], 0.0)   # RIGHT

# Test 3: one-hot trenutnog pravca
# Snake1 krece DESNO = Direction.RIGHT = (0,1) -> index 3 u [UP, DOWN, LEFT, RIGHT]
print("\n--- Test 3: trenutni pravac (bits [4:8]) ---")
dir_bits = state[4:8]
check("gore = 0",   dir_bits[0], 0.0)
check("dole = 0",   dir_bits[1], 0.0)
check("levo = 0",   dir_bits[2], 0.0)
check("desno = 1",  dir_bits[3], 1.0)

# Test 4: pravac hrane - rucno postavljanje hrane
print("\n--- Test 4: pravac hrane (bits [8:12]) ---")
game4 = Game()
game4.board.food = [(5, 5)]   # hrana dole-desno od glave (0,1)
s4 = get_state(game4.snake1, game4.snake2, game4.board)
food_bits = s4[8:12]
check("hrana je gore? -> NE",   food_bits[0], 0.0)
check("hrana je dole? -> DA",   food_bits[1], 1.0)
check("hrana je levo? -> NE",   food_bits[2], 0.0)
check("hrana je desno? -> DA",  food_bits[3], 1.0)

# Test 5: normalizovane duzine zmija (bits [12:14])
print("\n--- Test 5: normalizovane duzine (bits [12:14]) ---")
# Obe zmije imaju po 2 segmenta na pocetku; MAX_SNAKE_LENGTH = 100
s5 = get_state(game.snake1, game.snake2, game.board)
check("moja duzina = 2/100 = 0.02",       round(float(s5[12]), 4), 0.02)
check("protivnik duzina = 2/100 = 0.02",  round(float(s5[13]), 4), 0.02)

# Test 6: normalizovana udaljenost do protivnicke glave (bit [14])
print("\n--- Test 6: udaljenost do protivnika (bit [14]) ---")
# Glava snake1: (0,1), glava snake2: (9,8)
# Manhattan = |0-9| + |1-8| = 9 + 7 = 16; MAX_DISTANCE = 20
expected_dist = round(16 / 20, 4)
check(f"manhattan dist = 16/20 = {expected_dist}", round(float(s5[14]), 4), expected_dist)

# Test 7: space bits su u [0, 1]
print("\n--- Test 7: flood fill space (bits [15:19]) ---")
space_bits = s5[15:19]
check("svi space bits >= 0", bool(np.all(space_bits >= 0)), True)
check("svi space bits <= 1", bool(np.all(space_bits <= 1)), True)
print(f"         space vrednosti: {np.round(space_bits, 3)}")

# Test 8: _manhattan helper
print("\n--- Test 8: _manhattan ---")
check("(0,0)-(3,4) = 7", _manhattan((0,0), (3,4)), 7)
check("(5,5)-(5,5) = 0", _manhattan((5,5), (5,5)), 0)

# Test 9: _flood_fill u uglu
print("\n--- Test 9: _flood_fill (ugao table) ---")
game9 = Game()
# Iz ugla (0,0) sa praznom blociranom skupinom treba biti dostupno mnogo celija
free = _flood_fill((0, 0), game9.board, set())
check("flood fill iz ugla > 50", free > 50, True)
print(f"         dostupnih celija iz (0,0): {free}")

# Test 10: get_state je konzistentno (isti ulaz -> isti izlaz)
print("\n--- Test 10: determinizam ---")
game10 = Game()
state_a = get_state(game10.snake1, game10.snake2, game10.board)
state_b = get_state(game10.snake1, game10.snake2, game10.board)
check("dva poziva sa istim ulazom daju isti izlaz", np.array_equal(state_a, state_b), True)

print("\n" + "=" * 50)
print("Testovi zavrseni.")
print("=" * 50)
