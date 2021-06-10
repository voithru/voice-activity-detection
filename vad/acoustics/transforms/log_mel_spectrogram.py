import librosa
import numpy as np

from vad.acoustics.transforms.transform import Transform
from vad.data_models.audio_data import AudioData


class LogMelSpectrogramTransform(Transform):
    feature_size: int

    def __init__(self, n_fft: int, hop_ms: int, window_ms: int, n_mels: int):
        self.n_fft = n_fft
        self.hop_ms = hop_ms
        self.window_ms = window_ms
        self.n_mels = n_mels

        self.feature_size = n_mels

    def apply(self, audio_data: AudioData) -> np.array:
        hop_samples = int(self.hop_ms / 1000 * audio_data.sample_rate)
        window_samples = int(self.window_ms / 1000 * audio_data.sample_rate)

        feature = librosa.feature.melspectrogram(
            y=audio_data.audio,
            sr=audio_data.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=hop_samples,
            win_length=window_samples,
        )
        feature = np.log(feature + 1e-6)
        return feature
