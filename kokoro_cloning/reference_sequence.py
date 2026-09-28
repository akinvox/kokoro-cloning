"""Speaker identity and reference-audio conditioning."""

from __future__ import annotations

import torch
from torch import nn

from .balanced_mapper import BalancedReferenceVoicepackMapper


class PhoneReferenceAttention(nn.Module):
    def __init__(self, width=512, memory_width=192):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, memory_width)
        self.attention = nn.MultiheadAttention(
            memory_width, 4, batch_first=True, dropout=0.0
        )
        self.output = nn.Linear(memory_width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, phones, phone_mask, prompt):
        assert phones.ndim == 3 and phones.shape[1] == 512
        memory, memory_mask, _ = prompt
        q = self.query(self.norm(phones.transpose(1, 2)))
        attended, _ = self.attention(
            q, memory, memory, key_padding_mask=memory_mask, need_weights=False
        )
        delta = self.output(attended).masked_fill(phone_mask.unsqueeze(-1), 0)
        return phones + delta.transpose(1, 2)


class SequenceReferenceMapper(nn.Module):
    """Pooled identity and acoustic reference memory for native inference."""

    def __init__(self, base_table):
        super().__init__()
        self.pooled = BalancedReferenceVoicepackMapper(base_table)
        self.conv1 = nn.Conv1d(80, 192, 5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(192, 192, 5, stride=2, padding=2)
        self.norm = nn.LayerNorm(192)
        layer = nn.TransformerEncoderLayer(
            192,
            4,
            384,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.global_style = nn.Linear(384, 256)
        nn.init.zeros_(self.global_style.weight)
        nn.init.zeros_(self.global_style.bias)
        self.content = PhoneReferenceAttention()
        self.prosody = PhoneReferenceAttention()

    @staticmethod
    def mask(lengths, count):
        return torch.arange(count, device=lengths.device)[None, :] >= lengths[:, None]

    def forward(self, wavlm, sdec, spred, rows, prompt):
        coarse = self.pooled(wavlm, sdec, spred, rows)
        return coarse + self.global_style(prompt[2])
