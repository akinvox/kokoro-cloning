import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe_wave(wave: torch.Tensor) -> torch.Tensor:
    magnitude = wave.abs()
    compressed = 0.95 + 0.03 * torch.tanh((magnitude - 0.95) / 0.03)
    return torch.where(magnitude <= 0.95, wave, wave.sign() * compressed).clamp(
        -0.98, 0.98
    )


def make_alignment(duration: torch.Tensor) -> torch.Tensor:
    duration = duration.long().reshape(-1)
    require(torch.all(duration >= 1), "predicted duration contains a zero phone")
    total = int(duration.sum().item())
    require(0 < total <= 4000, f"unsafe predicted duration: {total}")
    alignment = torch.zeros(duration.numel(), total, device=duration.device)
    cursor = 0
    for phone_index, phone_duration in enumerate(duration.tolist()):
        alignment[phone_index, cursor : cursor + int(phone_duration)] = 1
        cursor += int(phone_duration)
    require(cursor == total, "predicted alignment cursor mismatch")
    return alignment.unsqueeze(0)
