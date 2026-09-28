"""Inference-only low-rank weight parametrizations; originals stay untouched."""

import torch
from torch import nn
from torch.nn.utils import parametrize


class Adapter(nn.Module):
    def __init__(self, a, b, shape, alpha):
        super().__init__()
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.shape = tuple(shape)
        self.scale = float(alpha) / a.shape[0]
        self.enabled = True

    def forward(self, original):
        if not self.enabled:
            return original
        return (
            original
            + (self.b @ self.a).reshape(self.shape).to(original.dtype) * self.scale
        )


def attach(core, targets, state):
    adapters = []
    consumed = set()
    for target in targets:
        module = core[target["root"]]
        if target["module"]:
            module = module.get_submodule(target["module"])
        name = target["parameter"]
        original = getattr(module, name)
        ak, bk = target["prefix"] + ".A", target["prefix"] + ".B"
        a, b = state[ak], state[bk]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != b.shape[1]
            or a.shape[0] != target["rank"]
            or b.shape[0] != original.shape[0]
            or a.shape[1] * b.shape[0] != original.numel()
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise ValueError("Invalid adapter tensor: " + target["prefix"])
        adapter = Adapter(a, b, original.shape, target["alpha"])
        parametrize.register_parametrization(module, name, adapter)
        adapters.append(adapter)
        consumed.update((ak, bk))
    if consumed != set(state) or len(adapters) != 199:
        raise ValueError("Adapter target inventory does not match this release")
    return adapters
