import os
from pathlib import Path
from typing import Dict, Optional

import torch

from vad.training.checkpointers.checkpointer import Checkpointer, ModelInfo, MonitorMode


class ModelCheckpointer(Checkpointer):
    checkpoints_dir: Path
    monitor_metric: str
    mode: MonitorMode
    top_k: int
    save_last: bool
    period: int
    name_format: str
    save_weights_only: bool
    extras: dict

    top_k_model_paths: Dict[Path, float]
    kth_best_checkpoint_path: Path
    kth_metric: float

    last_checkpoint_path: Optional[Path]

    def __init__(
        self,
        checkpoints_dir: Path,
        monitor_metric: str,
        mode: MonitorMode,
        top_k: int,
        save_last: bool,
        period: int,
        name_format: str,
        save_weights_only: bool,
        **kwargs,
    ):
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = checkpoints_dir
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.top_k = top_k
        self.save_last = save_last
        self.period = period
        self.name_format = name_format
        self.save_weights_only = save_weights_only
        self.extras = kwargs

        self.top_k_model_paths = {}
        self.kth_best_checkpoint_path = Path()
        self.kth_metric = 0
        self.last_checkpoint_path = None

    def checkpoint(self, model_info: ModelInfo, metrics: dict) -> None:
        if model_info.epoch % self.period != 0:
            return

        monitor_metric = self._get_monitor_metric(metrics)

        if self.last_checkpoint_path and self.last_checkpoint_path not in self.top_k_model_paths:
            self._delete_checkpoint(self.last_checkpoint_path)

        if self._in_top_k(monitor_metric):
            checkpoint_path = self._format_checkpoint_path(metrics)
            self._save_checkpoint(model_info, metrics, checkpoint_path)
            self._update_top_k(monitor_metric, checkpoint_path)
            self.last_checkpoint_path = checkpoint_path

        elif self.save_last:
            checkpoint_path = self._format_checkpoint_path(metrics)
            self._save_checkpoint(model_info, metrics, checkpoint_path)
            self.last_checkpoint_path = checkpoint_path

    def _get_monitor_metric(self, metrics: dict) -> float:
        return metrics[self.monitor_metric]

    def _in_top_k(self, metric: float) -> bool:
        if len(self.top_k_model_paths) < self.top_k:
            return True

        if self.mode == MonitorMode.MAX:
            return metric > self.kth_metric
        elif self.mode == MonitorMode.MIN:
            return metric < self.kth_metric

    def _update_top_k(self, metric: float, checkpoint_path: Path) -> None:
        if len(self.top_k_model_paths) == self.top_k:
            self.top_k_model_paths.pop(self.kth_best_checkpoint_path)
            self._delete_checkpoint(self.kth_best_checkpoint_path)
        self.top_k_model_paths[checkpoint_path] = metric

        self.kth_best_checkpoint_path, self.kth_metric = sorted(
            self.top_k_model_paths.items(), key=lambda x: x[1], reverse=self.mode == MonitorMode.MAX
        )[-1]

    def _save_checkpoint(self, model_info: ModelInfo, metrics: dict, path: Path) -> None:
        checkpoint_dict = {
            "state_dict": model_info.model.state_dict(),
            "epoch": model_info.epoch,
            "global_step": model_info.global_step,
            "monitor_metric": self.monitor_metric,
            "metrics": metrics,
            **self.extras,
        }
        if not self.save_weights_only:
            checkpoint_dict["optimizer_state_dict"] = model_info.optimizer.state_dict()
            checkpoint_dict["lr_scheduler_state_dict"] = model_info.lr_scheduler.state_dict()
            checkpoint_dict["grad_scaler_state_dict"] = model_info.grad_scaler.state_dict()
        torch.save(checkpoint_dict, str(path))

    @staticmethod
    def _delete_checkpoint(path: Path) -> None:
        os.remove(str(path))

    def _format_checkpoint_path(self, metrics: dict):
        checkpoint_name = self._format_checkpoint_name(metrics)
        checkpoint_path = self.checkpoints_dir.joinpath(checkpoint_name)
        return checkpoint_path

    def _format_checkpoint_name(self, metrics: dict) -> str:
        return f"{self.name_format.format(**metrics)}"
