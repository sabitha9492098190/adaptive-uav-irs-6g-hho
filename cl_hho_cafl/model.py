from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig


class GRULaneChangePredictor(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        dropout = config.dropout if config.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(sequence)
        pooled = encoded[:, -1, :]
        return self.classifier(pooled).squeeze(-1)


@dataclass(slots=True)
class LocalModelUpdate:
    client_id: int
    state_dict: dict[str, torch.Tensor]
    train_loss: float
    eval_loss: float
    eval_accuracy: float
    num_samples: int
    aggregation_weight: float


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)
