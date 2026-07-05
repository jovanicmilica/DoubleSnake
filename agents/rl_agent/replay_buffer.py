import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Cuva iskustva agenta u obliku (obs, action, reward, next_obs, done).
    DQN ne uci direktno iz svjezih iskustava vec uzima random batch
    iz ovog buffera — to lomi korelaciju izmedju uzastopnih koraka.
    """

    def __init__(self, capacity):
        # deque automatski brise najstariji element kad se popuni
        self._buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        """Dodaje jedno iskustvo u buffer."""
        self._buffer.append((
            np.array(obs,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        """
        Vraca random batch iskustava kao numpy arraye.
        Svaki od 5 arraya ima shape (batch_size, ...).
        """
        batch = random.sample(self._buffer, batch_size)

        obs, actions, rewards, next_obs, dones = zip(*batch)

        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),      # (B,)
            np.array(rewards,  dtype=np.float32),    # (B,)
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),    # (B,)  float jer mnozimo u loss-u
        )

    def __len__(self):
        return len(self._buffer)
