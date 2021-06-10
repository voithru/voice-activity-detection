from dataclasses import dataclass
from typing import Callable

import numpy as np

from vad.acoustics.spec_augment.random_resized_crop import RandomResizedCrop
from vad.acoustics.spec_augment.spec_augment import Compose, SpecAugment, UseWithProb


@dataclass
class SpecAugmentConfig:
    resize_scale_min: float = 0
    resize_scale_max: float = 1.0
    resize_ratio_min: float = 1.7
    resize_ratio_max: float = 2.3
    resize_prob: float = 0
    spec_num_mask: int = 0
    spec_freq_masking: float = 0
    spec_time_masking: float = 0
    spec_prob: float = 0


class SpecAugmentor:
    augment_function: Callable[[np.array], np.array]

    def __init__(self, config: SpecAugmentConfig):
        self.config = config

        self.augment_function = Compose(
            [
                UseWithProb(
                    RandomResizedCrop(
                        scale=(self.config.resize_scale_min, self.config.resize_scale_max),
                        ratio=(self.config.resize_ratio_min, self.config.resize_ratio_max),
                    ),
                    prob=self.config.resize_prob,
                ),
                UseWithProb(
                    SpecAugment(
                        num_mask=self.config.spec_num_mask,
                        freq_masking=self.config.spec_freq_masking,
                        time_masking=self.config.spec_time_masking,
                    ),
                    self.config.spec_prob,
                ),
            ]
        )

    def augment(self, feature: np.array) -> np.array:
        return self.augment_function(feature)
