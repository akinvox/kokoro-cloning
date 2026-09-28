#!/usr/bin/env python3
"""Reference mapper with independently normalized identity feature blocks."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F


ROWS = 510
STYLE_DIM = 256
WAVLM_DIM = 512
RAW_SDEC_DIM = 128
RAW_SPRED_DIM = 128
REFERENCE_DIM = WAVLM_DIM + RAW_SDEC_DIM + RAW_SPRED_DIM
HIDDEN_DIM = 384
LENGTH_FREQUENCIES = 16
LENGTH_DIM = 2 + 2 * LENGTH_FREQUENCIES
CONTRACT = "en-v15-balanced-reference-mapper-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def balanced_reference_features(
    wavlm: torch.Tensor,
    raw_sdec: torch.Tensor,
    raw_spred: torch.Tensor,
) -> torch.Tensor:
    """Give each frozen observation family exactly one unit of L2 energy."""

    require(wavlm.shape[-1] == WAVLM_DIM, "WavLM feature dimension drift")
    require(raw_sdec.shape[-1] == RAW_SDEC_DIM, "raw s_dec dimension drift")
    require(raw_spred.shape[-1] == RAW_SPRED_DIM, "raw s_pred dimension drift")
    values = (
        F.normalize(wavlm.float(), dim=-1, eps=1.0e-8),
        F.normalize(raw_sdec.float(), dim=-1, eps=1.0e-8),
        F.normalize(raw_spred.float(), dim=-1, eps=1.0e-8),
    )
    require(all(bool(torch.isfinite(value).all()) for value in values), "non-finite reference feature")
    return torch.cat(values, dim=-1)


def length_features(row_indices: torch.Tensor) -> torch.Tensor:
    require(row_indices.dtype in (torch.int32, torch.int64), "row indices must be integral")
    require(bool(((row_indices >= 0) & (row_indices < ROWS)).all()), "voicepack row out of range")
    x = row_indices.float() / float(ROWS - 1)
    frequencies = torch.arange(
        1,
        LENGTH_FREQUENCIES + 1,
        device=row_indices.device,
        dtype=torch.float32,
    )
    phase = math.pi * x.unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat(
        (
            x.unsqueeze(-1),
            x.mul(2.0).sub(1.0).square().unsqueeze(-1),
            torch.sin(phase),
            torch.cos(phase),
        ),
        dim=-1,
    )


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width * 2)
        self.linear2 = nn.Linear(width * 2, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.linear1(self.norm(value)))
        return value + self.linear2(hidden) * (2.0**-0.5)


class BalancedReferenceVoicepackMapper(nn.Module):
    """Maps reference features into the Kokoro 510-row style coordinate system."""

    def __init__(self, base_table: torch.Tensor) -> None:
        super().__init__()
        require(tuple(base_table.shape) == (ROWS, STYLE_DIM), "base table ABI drift")
        require(bool(torch.isfinite(base_table).all()), "base table contains non-finite values")
        self.register_buffer("base_table", base_table.detach().float().contiguous())
        self.reference_in = nn.Linear(REFERENCE_DIM, HIDDEN_DIM)
        self.reference_norm = nn.LayerNorm(HIDDEN_DIM)
        self.reference_blocks = nn.ModuleList([ResidualBlock(HIDDEN_DIM) for _ in range(2)])
        self.length_in = nn.Linear(LENGTH_DIM, HIDDEN_DIM)
        self.fusion_blocks = nn.ModuleList([ResidualBlock(HIDDEN_DIM) for _ in range(3)])
        self.output_norm = nn.LayerNorm(HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, STYLE_DIM)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_reference(self, features: torch.Tensor) -> torch.Tensor:
        require(features.ndim == 2 and features.shape[-1] == REFERENCE_DIM, "reference ABI drift")
        hidden = F.silu(self.reference_norm(self.reference_in(features)))
        for block in self.reference_blocks:
            hidden = block(hidden)
        return hidden

    def forward(
        self,
        wavlm: torch.Tensor,
        raw_sdec: torch.Tensor,
        raw_spred: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> torch.Tensor:
        require(
            wavlm.shape[0] == raw_sdec.shape[0] == raw_spred.shape[0] == row_indices.shape[0],
            "reference/row batch mismatch",
        )
        features = balanced_reference_features(wavlm, raw_sdec, raw_spred)
        hidden = self.encode_reference(features) + self.length_in(length_features(row_indices))
        for block in self.fusion_blocks:
            hidden = block(hidden)
        style = self.base_table[row_indices] + self.output(self.output_norm(hidden))
        require(bool(torch.isfinite(style).all()), "mapper produced non-finite style")
        return style


__all__ = [
    "BalancedReferenceVoicepackMapper",
    "CONTRACT",
    "balanced_reference_features",
    "state_sha256",
]
