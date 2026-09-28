import torch
import torchaudio
mean=-4
std=4
to_mel=torchaudio.transforms.MelSpectrogram(n_mels=80,n_fft=2048,win_length=1200,hop_length=300)

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-05 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor
