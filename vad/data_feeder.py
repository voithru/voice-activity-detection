import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader

from vad.acoustics.feature_extractor import FeatureExtractor
from vad.configs.train_config import TrainConfig
from vad.data_models.vad_data import VADDataList
from vad.datasets.resolution_map_dataset import ResolutionMapDataset
from vad.datasets.two_stage_iterable_dataset import TwoStageIterableDataset
from vad.datasets.utils import load_noise_paths
from vad.training.collate import variable_length_collate
from vad.training.feeder import Feeder

logger = logging.getLogger(__name__)


@dataclass
class DataFeeder(Feeder):
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    train_feature_extractor: FeatureExtractor
    val_feature_extractor: FeatureExtractor
    val_data_lengths: List[int]

    @classmethod
    def from_config(cls, config: TrainConfig):
        train_feature_extractor = FeatureExtractor(config.feature_extractor, use_spec_augment=True)
        val_feature_extractor = FeatureExtractor(config.feature_extractor, use_spec_augment=False)
        train_dataloader = DataFeeder.create_train_data_loader(config, train_feature_extractor)
        val_dataloader, val_data_lengths = DataFeeder.create_val_data_loader(
            config, val_feature_extractor
        )
        return cls(
            train_dataloader,
            val_dataloader,
            train_feature_extractor,
            val_feature_extractor,
            val_data_lengths,
        )

    @staticmethod
    def create_train_data_loader(
        config: TrainConfig, feature_extractor: FeatureExtractor
    ) -> DataLoader:
        assert (
            config.context_resolution.context_window_half_frames - 1
        ) % config.context_resolution.context_window_jump_frames == 0
        train_path = Path(config.train_val_dir).joinpath(config.train_path)
        if config.data_dir:
            data_dir = Path(config.data_dir)
        else:
            data_dir = train_path.parent
        train_data_list = VADDataList.load(train_path)
        if config.noise_injector is not None:
            noise_paths = load_noise_paths(
                Path(config.noise_injector.noise_path),
                Path(config.noise_injector.noise_data_dir)
                if config.noise_injector.noise_data_dir is not None
                else None,
            )
            noise_ratio = config.noise_injector.noise_ratio
            min_snr = config.noise_injector.min_snr
            max_snr = config.noise_injector.max_snr
        else:
            noise_paths = None
            noise_ratio = 0
            min_snr = 0
            max_snr = 0

        train_dataset = TwoStageIterableDataset(
            map_dataset=ResolutionMapDataset,
            data_list=train_data_list,
            data_dir=data_dir,
            chunk_size=config.dataset_chunk_size,
            num_workers=config.num_workers,
            noise_paths=noise_paths,
            noise_ratio=noise_ratio,
            min_snr=min_snr,
            max_snr=max_snr,
            feature_extractor=feature_extractor,
            context_window_half_frames=config.context_resolution.context_window_half_frames,
            context_window_jump_frames=config.context_resolution.context_window_jump_frames,
            context_window_shift_frames=config.context_resolution.context_window_shift_frames,
            expand_target=config.model.name.upper() in ("BDNN", "ACAM", "SELF-ATTENTION"),
            calculate_global_normalization_factor=False,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=0,
            collate_fn=variable_length_collate(train_dataset.variable_length_fields),
            pin_memory=True,
        )
        return train_dataloader

    @staticmethod
    def create_val_data_loader(
        config: TrainConfig, feature_extractor: FeatureExtractor
    ) -> Tuple[DataLoader, List[int]]:
        val_path = Path(config.train_val_dir).joinpath(config.val_path)
        if config.data_dir:
            data_dir = Path(config.data_dir)
        else:
            data_dir = val_path.parent
        val_data_list = VADDataList.load(val_path)
        val_dataset = ResolutionMapDataset(
            data_pairs=val_data_list.pairs,
            data_dir=data_dir,
            noise_paths=None,
            noise_ratio=None,
            min_snr=None,
            max_snr=None,
            feature_extractor=feature_extractor,
            context_window_half_frames=config.context_resolution.context_window_half_frames,
            context_window_jump_frames=config.context_resolution.context_window_jump_frames,
            expand_target=config.model.name.upper() in ("BDNN", "ACAM", "SELF-ATTENTION"),
            calculate_global_normalization_factor=False,
            num_workers=config.num_workers,
        )

        val_data_lengths = val_dataset.data_lengths

        logger.info(f"Val dataset size : {len(val_dataset)}")

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=variable_length_collate(val_dataset.variable_length_fields),
            pin_memory=True,
        )
        return val_dataloader, val_data_lengths
