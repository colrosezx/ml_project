"""
model.py

Здесь определяются разные архитектуры нейросетей для регрессии.
"""
import torch.nn as nn


class MLP1(nn.Module):
    """Простой MLP: 1 скрытый слой"""
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class MLP2(nn.Module):
    """Углублённый MLP: 2 скрытых слоя с Dropout"""
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
