import re
import matplotlib.pyplot as plt

LOG_FILE = "training_log.txt"   # Ime tvog log fajla


def parse_log(filepath):
    """
    Parsira log fajl i vadi podatke:
    - episode
    - win_rate (%)
    - agent_food
    - opponent_food
    - epsilon
    - heuristic_p (opciono)
    """
    episodes = []
    win_rates = []
    agent_foods = []
    opponent_foods = []
    epsilons = []
    heuristic_ps = []

    # Regex za linije koje počinju sa "Ep"
    pattern = re.compile(
        r"Ep\s+(?P<ep>\d+)\s+\|.*?food:\s+(?P<af>\d+\.\d+)/(?P<of>\d+\.\d+)\s+\|.*?epsilon:\s+(?P<eps>\d+\.\d+)\s+\|.*?win rate:\s+(?P<wr>\d+\.\d+)%\s+\|.*?heuristic_p:\s+(?P<hep>\d+\.\d+)"
    )

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                episodes.append(int(match.group("ep")))
                win_rates.append(float(match.group("wr")))
                agent_foods.append(float(match.group("af")))
                opponent_foods.append(float(match.group("of")))
                epsilons.append(float(match.group("eps")))
                heuristic_ps.append(float(match.group("hep")))

    return {
        "episodes": episodes,
        "win_rate": win_rates,
        "agent_food": agent_foods,
        "opponent_food": opponent_foods,
        "epsilon": epsilons,
        "heuristic_p": heuristic_ps,
    }


def plot_metrics(data):
    if not data["episodes"]:
        print("Nema podataka za crtanje. Proveri da li log fajl sadrži linije sa 'Ep'.")
        return

    episodes = data["episodes"]
    win_rate = data["win_rate"]
    epsilon = data["epsilon"]

    # Pretvori epsilon u procente (0–100 umesto 0–1)
    epsilon_percent = [e * 100 for e in epsilon]

    # Izračunaj udeo hrane (u procentima)
    food_ratio = []
    for af, of in zip(data["agent_food"], data["opponent_food"]):
        total = af + of
        if total == 0:
            food_ratio.append(50.0)   # nerešeno
        else:
            food_ratio.append(af / total * 100)

    # ----- Kreiraj grafik sa dve Y-ose -----
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Leva Y-osa: Win Rate (%) i Udeo hrane (%)
    ax1.set_xlabel("Epizoda", fontsize=12)
    ax1.set_ylabel("Procenti (Win Rate / Udeo hrane)", color="black", fontsize=12)
    
    # Win Rate (puna linija)
    color1 = "darkorange"
    ax1.plot(episodes, win_rate, color=color1, linewidth=2.5, label="Win Rate (%)")
    ax1.axhline(y=50, color=color1, linestyle=":", alpha=0.6, label="Slučajno (50%)")
    
    # Udeo hrane (isprekidana linija)
    color_food = "purple"
    ax1.plot(episodes, food_ratio, color=color_food, linewidth=2, linestyle="dashed", label="Udeo hrane (Agent) (%)")
    ax1.axhline(y=50, color=color_food, linestyle=":", alpha=0.4)
    
    ax1.tick_params(axis="y")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Desna Y-osa: Epsilon u procentima (0–100)
    ax2 = ax1.twinx()
    color_eps = "blue"
    ax2.set_ylabel("Epsilon (%)", color=color_eps, fontsize=12)
    ax2.plot(episodes, epsilon_percent, color=color_eps, linewidth=2, label="Epsilon (%)")
    ax2.tick_params(axis="y", labelcolor=color_eps)
    ax2.set_ylim(0, 100)   # epsilon je sada u procentima

    # Spajanje legendi sa obe ose
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.title("Win Rate, udeo hrane i epsilon kroz epizode", fontsize=14, fontweight="bold")

    # Dodatni tekst na dnu (krajnje vrednosti)
    final_wr = win_rate[-1] if win_rate else 0
    final_food = food_ratio[-1] if food_ratio else 0
    final_eps = epsilon_percent[-1] if epsilon_percent else 0
    textstr = (
        f"Krajnji Win Rate: {final_wr:.1f}%  |  "
        f"Krajnji udeo hrane (Agent): {final_food:.1f}%  |  "
        f"Krajnji epsilon: {final_eps:.1f}%"
    )
    plt.figtext(0.5, 0.01, textstr, wrap=True, horizontalalignment="center",
                fontsize=12, bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("winrate_food_epsilon_plot.png", dpi=150)
    plt.show()
    print("Grafik sačuvan kao 'winrate_food_epsilon_plot.png'")


if __name__ == "__main__":
    try:
        data = parse_log(LOG_FILE)
        plot_metrics(data)
    except FileNotFoundError:
        print(f"Fajl '{LOG_FILE}' nije pronađen. Proveri putanju.")