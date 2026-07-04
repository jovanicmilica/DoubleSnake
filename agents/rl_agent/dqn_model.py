import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Neuronska mreza koja prima feature vektor (19,) i vraca
    Q-vrijednost za svaku od 4 moguce akcije.

    Arhitektura: 19 → 256 → 256 → 128 → 4
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(19, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        """
        x: tensor oblika (batch_size, 19) ili (19,) za jedan primjer
        vraca: tensor oblika (batch_size, 4) — Q-vrijednosti za svaku akciju
        """
        return self.net(x)
