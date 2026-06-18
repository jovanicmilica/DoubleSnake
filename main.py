import sys
import io

from game.game import Game, GameState
from game.snake import Direction

# Test 1: osnovno kretanje
print("Pocetno stanje:")
game = Game()
print(f"Snake1 glava: {game.snake1.head()}")
print(f"Snake2 glava: {game.snake2.head()}")
print(f"Hrana: {game.board.food}")

for i in range(5):
    reward1, reward2, done = game.step(Direction.RIGHT, Direction.LEFT)
    print(f"\nKorak {i+1}:")
    print(f"  Snake1 glava: {game.snake1.head()}, tijelo: {game.snake1.body}")
    print(f"  Snake2 glava: {game.snake2.head()}, tijelo: {game.snake2.body}")
    print(f"  Reward1: {reward1}, Reward2: {reward2}")
    print(f"  Done: {done}")

    if done:
        print(f"  Pobjednik: {game.get_state()}")
        break

# Test 2: kolizija sa zidom
print("\n--- Test: kolizija sa zidom ---")
game2 = Game()
for i in range(15):
    reward1, reward2, done = game2.step(Direction.UP, Direction.LEFT)
    if done:
        print(f"Gotovo u koraku {i+1}")
        print(f"Snake1 alive: {game2.snake1.alive}")
        print(f"Rezultat: {game2.get_state()}")
        break

# Test 3: skupljanje hrane
print("\n--- Test: skupljanje hrane ---")
game3 = Game()
game3.board.food = (0, 2)
print(f"Hrana postavljena na: {game3.board.food}")
print(f"Duzina snake1 prije: {len(game3.snake1.body)}")
game3.step(Direction.RIGHT, Direction.LEFT)  # glava dolazi na (0,2) - jede hranu, grew=True
print(f"Duzina nakon jedenja: {len(game3.snake1.body)}")
game3.step(Direction.RIGHT, Direction.LEFT)  # ovaj move() ne radi pop() - zmija raste
print(f"Duzina nakon rasta: {len(game3.snake1.body)}")