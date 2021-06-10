from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class AdamConfig:
    eps: float = MISSING
    weight_decay_val: float = 0
