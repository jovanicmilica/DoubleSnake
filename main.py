"""
Double Snake — glavna ulazna tacka.

Pokreni: python main.py
"""


def menu():
    print("=" * 40)
    print("       DOUBLE SNAKE")
    print("=" * 40)
    print("  1. Igrac vs Heuristicki agent")
    print("  2. Heuristicki agent vs Heuristicki agent")
    print("  3. Treniraj DQN agenta")
    print("  4. Izlaz")
    print("=" * 40)
    return input("Odabir: ").strip()


def main():
    choice = menu()

    if choice == "1":
        from game_modes.player_vs_agent import run
        run()

    elif choice == "2":
        from game_modes.agent_vs_agent import run
        run()

    elif choice == "3":
        from agents.rl_agent.train import run
        run()

    elif choice == "4":
        print("Dovidjenja!")

    else:
        print("Nepoznat odabir.")
        main()


if __name__ == "__main__":
    main()
