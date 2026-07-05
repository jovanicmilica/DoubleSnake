"""
Iscrtava tok treninga iz checkpointa.

Kako radi: latest.pt sadrzi samo KUMULATIVNE brojeve (ukupne pobjede do tog
trenutka), pa se sama po sebi ne moze nacrtati kriva ucenja. Zato skripta
ucitava SVE ep_*.pt checkpointe (cuvaju se svakih 500 epizoda) i iz njih
rekonstruise:
  - kumulativni win rate kroz epizode
  - win rate PO INTERVALU (zadnjih 500 epizoda) - najbolje pokazuje napredak
  - opadanje epsilona
Ako postoje samo latest.pt ili jedan checkpoint, crta se finalna raspodjela
ishoda (bar chart).

Pokretanje iz korijena projekta:
    python plot_training.py
    python plot_training.py --dir checkpoints --out trening_grafikon.png
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import torch


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # stariji format / objekti koje weights_only ne prihvata
        return torch.load(path, map_location="cpu", weights_only=False)


def collect_history(ckpt_dir):
    """Ucita sve ep_N.pt checkpointe, sortirano po broju epizode."""
    paths = glob.glob(os.path.join(ckpt_dir, "ep_*.pt"))
    history = []
    for path in paths:
        m = re.search(r"ep_(\d+)\.pt$", os.path.basename(path))
        if not m:
            continue
        ckpt = load_checkpoint(path)
        history.append({
            "episode": ckpt.get("episode", int(m.group(1))),
            "epsilon": ckpt.get("epsilon"),
            "scores":  ckpt.get("scores", {}),
        })
    history.sort(key=lambda h: h["episode"])
    return history


def plot_history(history, out_path):
    episodes = [h["episode"] for h in history]
    wins1    = [h["scores"].get("player1", 0) for h in history]
    wins2    = [h["scores"].get("player2", 0) for h in history]
    draws    = [h["scores"].get("draws", 0)   for h in history]
    epsilons = [h["epsilon"] for h in history]

    # kumulativni win rate agenta
    cum_wr = [w / ep * 100 for w, ep in zip(wins1, episodes)]

    # win rate po intervalu (razlika izmedju susjednih checkpointa)
    int_ep, int_wr = [], []
    for i in range(1, len(history)):
        d_ep = episodes[i] - episodes[i - 1]
        d_w  = wins1[i] - wins1[i - 1]
        if d_ep > 0:
            int_ep.append(episodes[i])
            int_wr.append(d_w / d_ep * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- lijevi graf: win rate ---
    ax1.plot(episodes, cum_wr, color="#1f4e79", linewidth=2,
             label="kumulativni win rate")
    if int_wr:
        ax1.plot(int_ep, int_wr, color="#3fa34d", linewidth=1.2, alpha=0.85,
                 label="win rate po intervalu (500 ep)")
    ax1.axhline(50, color="#a83232", linestyle="--", linewidth=1,
                label="50% (izjednaceno)")
    ax1.set_xlabel("epizoda")
    ax1.set_ylabel("win rate agenta [%]")
    ax1.set_title("Napredak DQN agenta protiv heuristike")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- desni graf: epsilon ---
    if all(e is not None for e in epsilons):
        ax2.plot(episodes, epsilons, color="#e08c1a", linewidth=2)
        ax2.set_xlabel("epizoda")
        ax2.set_ylabel("epsilon")
        ax2.set_title("Opadanje istrazivanja (epsilon)")
        ax2.set_ylim(0, 1.05)
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Grafikon sacuvan: {out_path}")

    # finalna statistika u konzoli
    total = episodes[-1]
    print(f"\nFinalno stanje (epizoda {total}):")
    print(f"  Agent pobjede    : {wins1[-1]:6d} ({wins1[-1]/total*100:.1f}%)")
    print(f"  Heuristic pobjede: {wins2[-1]:6d} ({wins2[-1]/total*100:.1f}%)")
    print(f"  Nerijeseno       : {draws[-1]:6d} ({draws[-1]/total*100:.1f}%)")
    if int_wr:
        print(f"  Win rate u zadnjem intervalu: {int_wr[-1]:.1f}%")

    plt.show()


def plot_latest_only(ckpt_dir, out_path):
    """Fallback: samo latest.pt -> bar chart finalnih ishoda."""
    path = os.path.join(ckpt_dir, "latest.pt")
    ckpt = load_checkpoint(path)
    scores  = ckpt.get("scores", {})
    episode = ckpt.get("episode", 0)

    labels = ["agent (DQN)", "heuristika", "nerijeseno"]
    values = [scores.get("player1", 0), scores.get("player2", 0), scores.get("draws", 0)]
    colors = ["#1f4e79", "#a83232", "#999999"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for bar, v in zip(bars, values):
        pct = v / episode * 100 if episode else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v}\n({pct:.1f}%)", ha="center", va="bottom")
    ax.set_ylabel("broj partija")
    ax.set_title(f"Ishodi poslije {episode} epizoda treninga")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Grafikon sacuvan: {out_path}")
    print(f"epsilon = {ckpt.get('epsilon'):.3f}, koraka = {ckpt.get('steps')}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Iscrtava tok treninga iz checkpointa")
    parser.add_argument("--dir", default="checkpoints", help="folder sa checkpointima")
    parser.add_argument("--out", default="trening_grafikon.png", help="izlazni PNG")
    args = parser.parse_args()

    history = collect_history(args.dir)

    if len(history) >= 2:
        plot_history(history, args.out)
    elif os.path.exists(os.path.join(args.dir, "latest.pt")):
        print("Nadjen samo latest.pt (nema ep_*.pt istorije) - crtam finalne ishode.")
        plot_latest_only(args.dir, args.out)
    else:
        print(f"Nema checkpointa u '{args.dir}'.")


if __name__ == "__main__":
    main()
