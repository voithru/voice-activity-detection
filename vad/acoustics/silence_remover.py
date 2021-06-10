from dataclasses import dataclass
from datetime import timedelta

import librosa
import numpy as np
from omegaconf import MISSING

from vad.data_models.audio_data import AudioData


@dataclass
class SilenceRemoverConfig:
    silence_threshold_db: float = MISSING


class SilenceRemover:
    config: SilenceRemoverConfig

    def __init__(self, config: SilenceRemoverConfig):
        self.config = config

    def remove_silence(self, audio_data: AudioData):
        non_silence_indices = librosa.effects.split(
            audio_data.audio, top_db=self.config.silence_threshold_db
        )
        audio = np.concatenate([audio_data.audio[start:end] for start, end in non_silence_indices])
        silence_removed = AudioData(
            audio=audio,
            sample_rate=audio_data.sample_rate,
            duration=timedelta(seconds=len(audio) / audio_data.sample_rate),
        )
        return silence_removed
