import os

from agents.rl_agent.dqn_agent import DQNAgent


CHECKPOINT_PATH = os.path.join("checkpoints", "latest.pt")


def load_trained_agent(path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        return None, f"Nije pronadjen checkpoint: {path}"

    agent = DQNAgent()
    try:
        episode, _ = agent.load(path)
    except Exception as exc:
        return None, f"Checkpoint ne moze da se ucita: {exc}"

    agent.epsilon = 0.0
    agent.model.eval()
    return agent, f"Ucitavan RL agent iz epizode {episode}"
