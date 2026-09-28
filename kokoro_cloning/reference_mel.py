"""Reference mel features expected by the released acoustic encoders."""

import torch
import torchaudio

mean = -4
std = 4
# The checkpoint expects this filterbank setting even for 24 kHz input audio.
to_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-05 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor
