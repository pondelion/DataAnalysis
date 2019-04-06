import math
import torch
import torch.nn as nn
import numpy as np
from .. import DEVICE


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, output_dim=2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(True),
            nn.Linear(12, output_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def square_loss(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        x = torch.stack([torch.from_numpy(d).float().to(DEVICE) for d in x])

        reconst = self.forward(x).detach().cpu().numpy()
        x = np.array(x)

        return ((x - reconst)**2).sum(axis=1)


class CNNAutoEncoder(nn.Module):

    def __init__(self, input_dim=31, output_dim=2, in_channels=1):
        super(CNNAutoEncoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, 4, 2, 2),  # (B, 1, 32) → (B, 16, 16)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(16, 16*2, 4, 2, 1),  # (B, 16, 16) → (B, 32, 8)
            nn.BatchNorm1d(16*2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(16*2, 16*4, 4, 2, 1),  # (B, 32, 8) → (B, 64, 4)
            nn.BatchNorm1d(16*4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(16*4, 16*8, 4, 2, 1),  # (B, 64, 4) → (B, 128, 2)
            nn.BatchNorm1d(16*8),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(128*math.ceil(input_dim/16), output_dim),
            nn.Tanh()
        )

        self.decorder1 = nn.Sequential(
            nn.ConvTranspose1d(output_dim, 16*8, 4, 1, 0),  # (B, 2, 1) → (B, 16*8, 4)
            nn.BatchNorm1d(16*8),
            nn.ReLU(),
            nn.ConvTranspose1d(16*8, 16*4, 4, 2, 1),  # (B, 16*8, 4) → (B, 16*4, 8)
            nn.BatchNorm1d(16*4),
            nn.ReLU(),
            nn.ConvTranspose1d(16*4, 16*2, 4, 2, 1),  # (B, 16*4, 8) → (B, 16*2, 16)
            nn.BatchNorm1d(16*2),
            nn.ReLU(),
            nn.ConvTranspose1d(16*2, 16, 3, 2, 1),  # (B, 16*2, 16) → (B, 16, 32)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder1(x)
        x = x.view(x.shape[0], -1)
        x = self.encoder2(x)
        return x

    def decode(self, x):
        x = self.decorder1(x.unsqueeze(-1))
        return x

    def square_loss(self, x):
        if len(x.shape) == 1:
            # (N) → (B(1), C(1), N)
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=0)
        elif len(x.shape) == 2:
            # (B, N) → (B, C(1), N)
            x = np.expand_dims(x, axis=1)

        x = torch.stack([torch.from_numpy(d).float() for d in x])

        reconst = self.forward(x).detach().numpy()
        x = np.array(x)

        return ((x - reconst)**2)[:, 0, :].sum(axis=1)
