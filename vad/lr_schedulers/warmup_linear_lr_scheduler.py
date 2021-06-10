from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class WarmupLinearConfig:
    warmup_steps: int = MISSING
    total_steps: int = MISSING  # dataset_size // gradient_accumulation_steps * epochs
