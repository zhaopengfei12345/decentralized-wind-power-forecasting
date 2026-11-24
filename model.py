import torch
from torch import nn


class WindGRU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        head_dropout = 0.33719747955403395
        self.encoder = nn.GRU(
            input_dim, 128, num_layers=3, batch_first=True, dropout=0.2861252792497351
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 32), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(32, 32), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        sequence, _ = self.encoder(x)
        return torch.cat([sequence[:, -1], sequence.mean(dim=1)], dim=-1)

    def forward(self, x):
        return self.head(self.encode(x))
