from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import librosa
import numpy as np
import soundfile

STANDARD_SAMPLE_RATE = 16000


@dataclass
class AudioData:
    audio: np.array  # 1D array of samples
    sample_rate: int
    duration: timedelta

    @classmethod
    def load(cls, path: Path):
        if path.suffix == ".pcm":
            with path.open("rb") as audio_file:
                # Convert 16 bit signed (-32768, +32767) to float
                audio = np.fromfile(audio_file, dtype=np.int16).astype(np.single) / 32768
        else:
            audio, sample_rate = soundfile.read(path, dtype=np.single, always_2d=True)
            audio = audio.mean(axis=1)
            if sample_rate != STANDARD_SAMPLE_RATE:
                audio = librosa.resample(
                    audio, sample_rate, STANDARD_SAMPLE_RATE, res_type="kaiser_fast"
                )

        duration = timedelta(seconds=len(audio) / STANDARD_SAMPLE_RATE)
        audio_data = cls(audio=audio, sample_rate=STANDARD_SAMPLE_RATE, duration=duration)
        return audio_data

    def save(self, path: Path):
        soundfile.write(path, self.audio, self.sample_rate)
