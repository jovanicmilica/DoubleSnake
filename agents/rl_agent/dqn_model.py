import torch
import torch.nn as nn

from game.state_utils import STATE_SIZE


class DQN(nn.Module):
    """
    Neuronska mreza koja prima feature vektor i vraca
    Q-vrijednost za svaku od 4 moguce akcije.

    Arhitektura: STATE_SIZE -> 256 -> 256 -> 128 -> 4
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        """
        x: tensor oblika (batch_size, STATE_SIZE) ili (STATE_SIZE,) za jedan primjer
        vraca: tensor oblika (batch_size, 4) — Q-vrijednosti za svaku akciju
        """
        return self.net(x)
