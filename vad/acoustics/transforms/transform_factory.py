from dataclasses import dataclass
from enum import Enum
from typing import Optional

from omegaconf import MISSING

from vad.acoustics.transforms.log_mel_spectrogram import LogMelSpectrogramTransform
from vad.acoustics.transforms.mel_spectrogram import MelSpectrogramTransform
from vad.acoustics.transforms.mfcc import MFCCTransform
from vad.acoustics.transforms.spectrogram import SpectrogramTransform


class TransformName(Enum):
    Spectrogram = "spectrogram"
    MelSpectrogram = "mel"
    LogMelSpectrogramTransform = "log-mel"
    MFCC = "mfcc"


@dataclass
class TransformConfig:
    name: str = MISSING
    n_fft: int = MISSING
    hop_ms: int = MISSING
    window_ms: int = MISSING
    n_mels: Optional[int] = None
    n_mfcc: Optional[int] = None


def create_transform(config: TransformConfig):
    name = TransformName(config.name)
    if name == TransformName.Spectrogram:
        return SpectrogramTransform(
            n_fft=config.n_fft, hop_ms=config.hop_ms, window_ms=config.window_ms
        )
    elif name == TransformName.MelSpectrogram:
        return MelSpectrogramTransform(
            n_fft=config.n_fft,
            hop_ms=config.hop_ms,
            window_ms=config.window_ms,
            n_mels=config.n_mels,
        )
    elif name == TransformName.LogMelSpectrogramTransform:
        return LogMelSpectrogramTransform(
            n_fft=config.n_fft,
            hop_ms=config.hop_ms,
            window_ms=config.window_ms,
            n_mels=config.n_mels,
        )
    elif name == TransformName.MFCC:
        return MFCCTransform(
            n_fft=config.n_fft,
            hop_ms=config.hop_ms,
            window_ms=config.window_ms,
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
        )
    else:
        raise NotImplementedError
