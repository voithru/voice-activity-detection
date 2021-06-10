from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from vad.training.checkpointers.checkpointer import Checkpointer, ModelInfo
from vad.training.feeder import Feeder
from vad.training.loggers.logger import Logger
from vad.training.progress_bar import ProgressBar
from vad.training.runner import Runner
from vad.training.training_info import TrainingInfo
from vad.training.utils import dictionarize_list, to_device, to_float, to_numpy


class Trainer:
    logger: Logger
    model_checkpoint: Checkpointer
    progress_bar: ProgressBar

    gradient_accumulation_steps: int
    gradient_clip_val: Optional[float]
    epochs: Optional[int]
    # steps: Optional[int]
    device: torch.device
    resume_from_checkpoint: Optional[Path]
    reset_lr_scheduling: bool

    use_amp: bool
    grad_scaler: torch.cuda.amp.GradScaler
    previous_scale = float

    epoch: int
    global_step: int
    total_num_batches: int

    model: nn.Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer: Optimizer
    lr_scheduler: _LRScheduler
    model_runner: Runner

    def __init__(
        self,
        logger: Logger,
        model_checkpoint: Checkpointer,
        progress_bar: ProgressBar,
        gradient_accumulation_steps: int,
        gradient_clip_val: Optional[float],
        epochs: int,
        # steps: int,  # TODO
        num_sanity_check_steps: int,
        device: torch.device,
        use_amp: bool,
        resume_from_checkpoint: Optional[Path] = None,
        reset_lr_scheduling: bool = False,
    ):
        self.logger = logger
        self.model_checkpoint = model_checkpoint
        self.progress_bar = progress_bar

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.epochs = epochs
        # self.steps = steps
        self.num_sanity_check_steps = num_sanity_check_steps
        self.device = device
        self.resume_from_checkpoint = resume_from_checkpoint
        self.reset_lr_scheduling = reset_lr_scheduling

        self.use_amp = use_amp
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.previous_scale = self.grad_scaler.get_scale()
        # TODO: Saving & Loading grad_scaler

        self.epoch = 0
        self.global_step = 0

    def train(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        data_feeder: Feeder,
        model_runner: Runner,
    ):
        self.initialize_trainer(model, optimizer, lr_scheduler, data_feeder, model_runner)
        self.run()

    def initialize_trainer(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        data_feeder: Feeder,
        model_runner: Runner,
    ):
        if self.resume_from_checkpoint:
            checkpoint = torch.load(self.resume_from_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
            if not self.reset_lr_scheduling:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device=self.device)
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            self.epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_dataloader = data_feeder.train_dataloader
        self.val_dataloader = data_feeder.val_dataloader

        self.model_runner = model_runner

    def run(self):
        self.sanity_check()
        self.run_epochs(self.epochs)

    def sanity_check(self):
        self.progress_bar.on_sanity_check_start(num_sanity_check_steps=self.num_sanity_check_steps)

        self.model.eval()
        val_results = []
        for batch_index, batch in enumerate(self.val_dataloader):
            if batch_index >= self.num_sanity_check_steps:
                break
            with torch.no_grad():
                val_result = self.run_val_step(batch_index, batch)
            val_results.append(to_numpy(val_result))
        val_results_dict = dictionarize_list(val_results)
        val_result_aggregated = self.model_runner.validation_epoch_end(val_results_dict)
        val_result_float = to_float(val_result_aggregated)
        self.progress_bar.on_validation_end(val_result_float)
        self.progress_bar.on_sanity_check_end()

    def run_epochs(self, epochs: int):
        self.progress_bar.on_train_start()

        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            self.run_epoch()

        self.progress_bar.on_train_end()

    def run_epoch(self):
        self.progress_bar.on_epoch_start(epoch=self.epoch)

        self.run_train_epoch()
        val_result_aggregated = self.run_val_epoch()

        self.checkpoint(val_result_aggregated)
        self.progress_bar.on_epoch_end()

    def run_train_epoch(self):
        self.model.train()
        for batch_index, batch in enumerate(self.train_dataloader):
            self.run_train_step(batch_index, batch)

    def run_train_step(self, batch_index, batch):
        batch = to_device(batch, self.device)
        info = TrainingInfo(self.epoch, self.global_step, batch_index)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            train_result = self.model_runner.training_step(self.model, batch, info)
            loss = train_result["loss"]

        # Accumulate loss
        loss = loss / self.gradient_accumulation_steps

        self.grad_scaler.scale(loss).backward()

        accumulation_done = (batch_index + 1) % self.gradient_accumulation_steps == 0
        try:
            is_final_batch = (batch_index + 1) == len(self.train_dataloader)
        except TypeError:  # IterableDataset
            is_final_batch = False
        if accumulation_done or is_final_batch:
            if self.gradient_clip_val is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()  # TODO: set_to_none=True
            if self.grad_scaler.get_scale() == self.previous_scale:
                self.lr_scheduler.step()
            else:
                self.previous_scale = self.grad_scaler.get_scale()
            self.global_step += 1

        train_result = to_float(train_result)
        train_result["epoch"] = self.epoch
        train_result["global_step"] = self.global_step
        train_result["lr"] = self.optimizer.param_groups[0]["lr"]
        self.progress_bar.on_batch_end(train_result)
        try:
            self.progress_bar.main_progress_bar.total = len(self.train_dataloader) + len(
                self.val_dataloader
            )
        except TypeError:  # IterableDataset
            pass
        self.progress_bar.main_progress_bar.refresh()
        self.logger.log_metrics_at_intervals(train_result)

    def run_val_epoch(self):
        self.progress_bar.on_validation_start(val_num_batches=len(self.val_dataloader))
        val_results = []
        self.model.eval()

        for batch_index, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                val_result = self.run_val_step(batch_index, batch)
            val_results.append(to_numpy(val_result))
        val_results_dict = dictionarize_list(val_results)
        val_metrics = self.model_runner.validation_epoch_end(val_results_dict)
        val_metrics = to_float(val_metrics)

        val_metrics["epoch"] = self.epoch
        val_metrics["global_step"] = self.global_step
        self.progress_bar.on_validation_end(val_metrics)
        self.logger.log_metrics(val_metrics)
        return val_metrics

    def run_val_step(self, batch_index, batch):
        batch = to_device(batch, self.device)
        info = TrainingInfo(self.epoch, self.global_step, batch_index)
        val_result = self.model_runner.validation_step(self.model, batch, info)
        self.progress_bar.on_validation_batch_end()
        if not self.progress_bar.main_progress_bar.disable:
            try:
                self.progress_bar.main_progress_bar.total = len(self.train_dataloader) + len(
                    self.val_dataloader
                )
            except TypeError:  # IterableDataset
                pass
            self.progress_bar.main_progress_bar.refresh()
        return val_result

    def checkpoint(self, val_metrics):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        self.model_checkpoint.checkpoint(
            model_info=ModelInfo(
                model=model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                grad_scaler=self.grad_scaler,
                epoch=self.epoch,
                global_step=self.global_step,
            ),
            metrics=val_metrics,
        )
