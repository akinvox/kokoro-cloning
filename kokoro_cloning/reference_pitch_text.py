import math

import torch
import torch.nn.functional as F
from torch import nn


def positions(count, width, device, dtype):
    pos = torch.arange(count, device=device, dtype=dtype)[:, None]
    freq = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / width)
    )
    return torch.stack((torch.sin(pos * freq), torch.cos(pos * freq)), -1).flatten(-2)


class ReferenceTranscript(nn.Module):
    """Enrollment-only reference phonemes -> acoustic memory residual."""

    def __init__(self, vocabulary=178, width=192):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)
        self.text_norm = nn.LayerNorm(width)
        self.acoustic_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, acoustic, acoustic_mask, phones, phone_mask):
        assert phones.ndim == 2 and phone_mask.shape == phones.shape
        assert phones.dtype == torch.long and phone_mask.dtype == torch.bool
        assert not bool(phone_mask.all(1).any()), "Empty reference transcript"
        assert len(phones) == len(acoustic)
        assert bool(((phones >= 0) & (phones < self.embedding.num_embeddings)).all())
        text = self.embedding(phones)
        text = self.text_norm(
            text + positions(text.shape[1], text.shape[2], text.device, text.dtype)
        )
        (delta, _) = self.attention(
            self.acoustic_norm(acoustic),
            text,
            text,
            key_padding_mask=phone_mask,
            need_weights=False,
        )
        delta = self.output(delta).masked_fill(acoustic_mask[..., None], 0)
        return acoustic + delta


def attach_reference_text(shared):
    """Initialize transcript conditioning without changing the caller’s random state."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20260917)
        shared.reference_transcript = ReferenceTranscript()
    return shared


def encode_reference(shared, mel, lengths, phones=None, phone_mask=None):
    """Encode reference audio and its transcript into acoustic memory."""
    assert phones is not None and phone_mask is not None, "Reference transcript missing"
    assert mel.ndim == 3 and mel.shape[1] == 80
    assert bool((lengths > 0).all()) and bool((lengths <= mel.shape[-1]).all())
    mask = shared.mask(lengths, mel.shape[-1])
    x = F.gelu(shared.conv1(mel.masked_fill(mask[:, None], 0)))
    lengths = (lengths + 1) // 2
    x = x.masked_fill(shared.mask(lengths, x.shape[-1])[:, None], 0)
    x = F.gelu(shared.conv2(x))
    lengths = (lengths + 1) // 2
    mask = shared.mask(lengths, x.shape[-1])
    x = shared.norm(x.transpose(1, 2))
    x = shared.reference_transcript(x, mask, phones, phone_mask)
    x = shared.sequence(
        x + positions(x.shape[1], 192, x.device, x.dtype)[None],
        src_key_padding_mask=mask,
    )
    x = x.masked_fill(mask[..., None], 0)
    denom = lengths[:, None].to(x.dtype)
    mean = x.sum(1) / denom
    variance = (x - mean[:, None]).square().masked_fill(mask[..., None], 0).sum(
        1
    ) / denom
    stats = torch.cat((mean, variance.clamp_min(1e-08).sqrt()), -1)
    return (x, mask, stats)
