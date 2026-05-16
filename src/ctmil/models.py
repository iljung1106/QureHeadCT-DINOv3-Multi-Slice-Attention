from __future__ import annotations

import torch
from torch import nn


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.w(self.v(x) * self.u(x)).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class DINOv3BiGRUGatedABMIL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_labels: int,
        gru_layers: int = 1,
        attention_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bigru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        context_dim = hidden_dim * 2
        self.attention = GatedAttention(context_dim, attention_dim, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(context_dim), nn.Dropout(dropout), nn.Linear(context_dim, num_labels))

    def forward(self, features: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.proj(features)
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bigru(packed)
        contextual, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        bag, attention = self.attention(contextual, mask)
        logits = self.classifier(bag)
        return {"logits": logits, "attention": attention}

