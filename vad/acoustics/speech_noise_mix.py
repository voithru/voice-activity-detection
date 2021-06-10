import random
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import soundfile
from omegaconf import MISSING


@dataclass
class NoiseInjectorConfig:
    noise_path: str = MISSING
    noise_data_dir: Optional[str] = None
    noise_ratio: float = MISSING
    min_snr: float = MISSING
    max_snr: float = MISSING


def mix_speech_noise(
    speech_path,
    noise_paths,
    noise_ratio=1.0,
    min_snr=-10,
    max_snr=12,
    remove_silent_noise=True,
    silence_threshold=30,
    hop_samples=512,
):
    speech, sample_rate = soundfile.read(speech_path)
    noisy_speech = np.copy(speech)

    random.shuffle(noise_paths)
    index = 0
    for noise_path in noise_paths:
        noise, _ = soundfile.read(noise_path)
        if index + len(noise) > len(speech):
            noise = noise[: len(speech) - index]

        if remove_silent_noise and len(noise) >= hop_samples:
            non_silence_indices = librosa.effects.split(
                noise, top_db=silence_threshold, hop_length=hop_samples
            )
            noise = np.concatenate([noise[start:end] for start, end in non_silence_indices])

        noise_length = len(noise)

        snr = np.random.uniform(min_snr, max_snr)

        speech_segment = speech[index : index + noise_length]
        noisy_speech[index : index + noise_length] = add_noise(
            signal=speech_segment, noise=noise, snr=snr
        )

        index += noise_length

        # Add silence
        if 0.0 < noise_ratio < 1.0:
            silence_length = noise_length * (1 - noise_ratio) / noise_ratio
            index += int(silence_length)

        if index >= len(speech):
            break

    return noisy_speech


def add_noise(signal, noise, snr, epsilon=1e-8):
    signal_power = power(signal)
    noise_power = power(noise) + epsilon

    scale_factor = (signal_power / noise_power) * 10 ** (-snr / 10)

    scaled_noise = np.sqrt(scale_factor) * noise

    noisy_signal = signal + scaled_noise
    return noisy_signal


def power(samples):
    return np.mean(np.abs(samples) ** 2)


def remove_silence(audio, silence_threshold=30):
    non_silence_indices = librosa.effects.split(audio, top_db=silence_threshold)
    return np.concatenate([audio[start:end] for start, end in non_silence_indices])
