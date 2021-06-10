from abc import ABC

import numpy as np

from vad.data_models.audio_data import AudioData


class Transform(ABC):
    feature_size: int

    def apply(self, audio_data: AudioData) -> np.array:
        pass
