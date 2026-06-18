from game.game import Game, GameState
from game.snake import Direction

game = Game()

print("Pocetno stanje:")
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