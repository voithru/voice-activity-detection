from pathlib import Path
from typing import List, Optional

import torch
import typer
from omegaconf import OmegaConf

from vad.configs.train_config import TrainConfig
from vad.data_feeder import DataFeeder
from vad.lr_schedulers.lr_scheduler_factory import create_lr_scheduler
from vad.model_runner import ModelRunner
from vad.models.model_factory import create_model
from vad.optimizers.optimizer_factory import create_optimizer
from vad.training.checkpointers.checkpointer import MonitorMode
from vad.training.checkpointers.model_checkpointer import ModelCheckpointer
from vad.training.loggers.file_logger import FileLogger
from vad.training.progress_bar import ProgressBar
from vad.training.trainer import Trainer
from vad.training.utils import find_next_version_dir
from vad.util.seed import seed_everything


def train_vad_from_scratch(
    config_path: Path = typer.Argument(
        ...,
        help="Path to config yaml file. \t\t\t\t\t See violin/vad/configs/train_config.py for format definition. "
        + OmegaConf.to_yaml(OmegaConf.structured(TrainConfig)),
    ),
    set: Optional[List[str]] = typer.Option(
        None,
        help="Specify overrides of config on the command line. \t\t (e.g. --set field1=value1 --set field2=value2)",
    ),
):
    config: TrainConfig
    config = OmegaConf.structured(TrainConfig)
    config = OmegaConf.merge(config, OmegaConf.load(config_path))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(set))

    seed_everything(config.random_seed)

    assert (
        config.context_resolution.context_window_half_frames - 1
    ) % config.context_resolution.context_window_jump_frames == 0

    context_window_frames = (
        2
        * (config.context_resolution.context_window_half_frames - 1)
        // config.context_resolution.context_window_jump_frames
        + 3
    )

    data_feeder: DataFeeder = DataFeeder.from_config(config)
    model_runner = ModelRunner(config, context_window_frames)

    model = create_model(
        config.model, data_feeder.train_feature_extractor.feature_size, context_window_frames
    )
    optimizer = create_optimizer(model.parameters(), config.optimizer)
    lr_scheduler = create_lr_scheduler(optimizer, config.lr_scheduler)

    version_dir = find_next_version_dir(runs_dir=Path(config.runs_dir), run_name=config.run_name)

    model_logger = FileLogger(log_dir=version_dir)
    model_logger.save_config(OmegaConf.to_container(config))

    name_prefix = f"{config.run_name.replace('/', '-')}-{version_dir.name}-"
    model_checkpoint = ModelCheckpointer(
        checkpoints_dir=version_dir.joinpath("checkpoints"),
        monitor_metric="val_accuracy",
        mode=MonitorMode.MIN,
        top_k=1,
        save_last=True,
        period=1,
        name_format=name_prefix + "epoch-{epoch:0>3}-val-acc-{val_accuracy:.5f}.checkpoint",
        save_weights_only=False,
        config=OmegaConf.to_container(config),
    )

    progress_bar = ProgressBar(
        train_monitor_metrics=["loss", "lr", "acc"],
        val_monitor_metrics=["val_loss", "val_accuracy", "val_auc", "val_recall"],
        version=version_dir.name,
        refresh_rate=1,
    )

    trainer = Trainer(
        logger=model_logger,
        model_checkpoint=model_checkpoint,
        progress_bar=progress_bar,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_clip_val=config.gradient_clip_val,
        epochs=config.epochs,
        # steps=config.steps,
        num_sanity_check_steps=config.num_sanity_check_steps,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        use_amp=config.use_amp,
        resume_from_checkpoint=Path(config.resume_from_checkpoint)
        if config.resume_from_checkpoint
        else None,
        reset_lr_scheduling=config.reset_lr_scheduling,
    )
    trainer.train(model, optimizer, lr_scheduler, data_feeder, model_runner)
