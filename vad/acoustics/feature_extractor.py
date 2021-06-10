import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import librosa
import numpy as np
from joblib.hashing import Hasher
from omegaconf import MISSING, OmegaConf

from vad.acoustics.silence_remover import SilenceRemover, SilenceRemoverConfig
from vad.acoustics.spec_augment.spec_augmentor import SpecAugmentConfig, SpecAugmentor
from vad.acoustics.transforms.transform import Transform
from vad.acoustics.transforms.transform_factory import TransformConfig, create_transform
from vad.data_models.audio_data import AudioData


@dataclass
class FeatureExtractorConfig:
    silence_remover: Optional[SilenceRemoverConfig] = None
    transform: TransformConfig = TransformConfig()
    spec_augment: Optional[SpecAugmentConfig] = None
    temporal_differences: bool = MISSING
    stack_differences: bool = MISSING
    cachedir: Optional[str] = None


class FeatureExtractor:
    feature_size: int
    feature_depth: int

    silence_remover: Optional[SilenceRemover]
    transform: Transform
    spec_augmentor: Optional[SpecAugmentor]
    config: FeatureExtractorConfig

    cachedir: Optional[Path]

    def __init__(self, config: FeatureExtractorConfig, use_spec_augment: bool):
        if config.silence_remover:
            self.silence_remover = SilenceRemover(config.silence_remover)
        else:
            self.silence_remover = None
        self.transform = create_transform(config.transform)
        if config.spec_augment and use_spec_augment:
            self.spec_augmentor = SpecAugmentor(config.spec_augment)
        else:
            self.spec_augmentor = None
        self.config = config

        self.feature_size, self.feature_depth = self._calculate_feature_size_and_depth()

        if config.cachedir:
            self.cachedir = Path(config.cachedir)
            self.cachedir.mkdir(parents=True, exist_ok=True)
        else:
            self.cachedir = None

    def extract_from_path_with_postprocessing(self, audio_path: Path) -> np.array:
        feature = self.extract_from_path_with_cache(audio_path)
        if self.spec_augmentor:
            feature = self.spec_augmentor.augment(feature).squeeze()

        features = self._apply_temporal_differences(feature)

        # (feature_size, time_size, [stack_size]) -> (time_size, feature_size, [stack_size])
        features = np.swapaxes(features, 0, 1)
        return features

    def extract_with_postprocessing(self, audio_data: AudioData) -> np.array:
        feature = self.extract(audio_data)
        if self.spec_augmentor:
            feature = self.spec_augmentor.augment(feature).squeeze()

        features = self._apply_temporal_differences(feature)

        # (feature_size, time_size, [stack_size]) -> (time_size, feature_size, [stack_size])
        features = np.swapaxes(features, 0, 1)
        return features

    def extract_from_path_with_cache(self, audio_path: Path) -> np.array:
        if self.cachedir is None:
            return self.extract_from_path(audio_path)

        hasher = Hasher()
        argument_repr = json.dumps(
            (
                str(audio_path),
                OmegaConf.to_container(self.config.silence_remover)
                if self.config.silence_remover
                else None,
                OmegaConf.to_container(self.config.transform),
            ),
            sort_keys=True,
            ensure_ascii=False,
        )
        argument_hash = hasher.hash(argument_repr)
        cache_path = self.cachedir.joinpath(argument_hash)
        if not cache_path.exists():
            feature = self.extract_from_path(audio_path)
            joblib.dump(feature, cache_path)
        else:
            feature = joblib.load(cache_path)

        return feature

    def extract_from_path(self, audio_path: Path) -> np.array:
        audio_data = AudioData.load(audio_path)
        return self.extract(audio_data)

    def extract(self, audio_data: AudioData) -> np.array:
        print("audio_data", audio_data.duration)

        if self.silence_remover:
            audio_data = self.silence_remover.remove_silence(audio_data)
        feature = self.transform.apply(audio_data)
        print("feature", feature.shape)

        return feature

    def _calculate_feature_size_and_depth(self):
        if self.config.temporal_differences and self.config.stack_differences:
            feature_size = self.transform.feature_size
            feature_depth = 3
        elif self.config.temporal_differences and not self.config.stack_differences:
            feature_size = self.transform.feature_size * 3
            feature_depth = 1
        else:
            feature_size = self.transform.feature_size
            feature_depth = 1

        return feature_size, feature_depth

    def _apply_temporal_differences(self, feature: np.array) -> np.array:
        if self.config.temporal_differences:
            feature_delta = librosa.feature.delta(feature, axis=1, width=9)
            feature_delta_delta = librosa.feature.delta(feature, axis=1, width=9, order=2)

            feature_list = [feature, feature_delta, feature_delta_delta]

            if self.config.stack_differences:
                feature = np.stack(feature_list, axis=2)
            else:
                feature = np.concatenate(feature_list, axis=0)

        return feature
