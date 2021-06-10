from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from vad.acoustics.feature_extractor import FeatureExtractorConfig
from vad.acoustics.speech_noise_mix import NoiseInjectorConfig
from vad.configs.dataset_config import ContextResolutionConfig
from vad.configs.model_config import ModelConfig
from vad.lr_scheduling import LRSchedulerConfig
from vad.optimizing import OptimizerConfig


@dataclass
class TrainConfig:
    train_val_dir: str = "."
    train_path: str = MISSING
    val_path: str = MISSING
    data_dir: Optional[str] = None
    runs_dir: str = "results/runs"
    run_name: str = MISSING
    context_resolution: ContextResolutionConfig = ContextResolutionConfig()
    dataset_chunk_size: Optional[int] = None
    noise_injector: Optional[NoiseInjectorConfig] = None
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    lr_scheduler: Optional[LRSchedulerConfig] = None
    gradient_clip_val: Optional[float] = None
    gradient_accumulation_steps: int = 1
    batch_size: int = MISSING
    epochs: int = MISSING
    log_interval: int = MISSING
    check_val_every_n_epoch: int = MISSING
    num_sanity_check_steps: int = 3
    random_seed: int = MISSING
    num_workers: int = MISSING
    use_amp: bool = False
    resume_from_checkpoint: Optional[str] = None
    reset_lr_scheduling: bool = False
