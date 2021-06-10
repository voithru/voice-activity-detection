import numpy as np
import torch

from vad.acoustics.transforms.transform import Transform
from vad.data_models.audio_data import AudioData


class SpectrogramTransform(Transform):
    feature_size: int

    def __init__(self, n_fft: int, hop_ms: int, window_ms: int):
        self.n_fft = n_fft
        self.hop_ms = hop_ms
        self.window_ms = window_ms

        self.feature_size = n_fft // 2 + 1

    def apply(self, audio_data: AudioData) -> np.array:
        hop_samples = int(self.hop_ms / 1000 * audio_data.sample_rate)
        window_samples = int(self.window_ms / 1000 * audio_data.sample_rate)
        stft = torch.stft(
            torch.FloatTensor(audio_data.audio),
            self.n_fft,
            hop_length=hop_samples,
            win_length=window_samples,
            window=torch.hamming_window(window_samples),
            center=False,
            normalized=False,
            onesided=True,
        )

        stft = (stft[:, :, 0].pow(2) + stft[:, :, 1].pow(2)).pow(0.5)
        feature = stft.numpy()
        return feature
