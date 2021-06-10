import sys
from typing import List

from tqdm import tqdm


class ProgressBar:  # TODO: Refactor
    train_monitor_metrics: List[str]
    val_monitor_metrics: List[str]
    main_progress_bar: tqdm = None
    val_progress_bar: tqdm = None
    test_progress_bar: tqdm = None
    _train_batch_index: int
    _val_batch_index: int
    is_disabled: bool

    def __init__(
        self,
        train_monitor_metrics: List[str],
        val_monitor_metrics: List[str],
        version: str,
        refresh_rate: int,
    ):
        self.train_monitor_metrics = train_monitor_metrics
        self.val_monitor_metrics = val_monitor_metrics
        self.version = version
        self.refresh_rate = refresh_rate
        self._train_batch_index = 0
        self._val_batch_index = 0

        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def init_sanity_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Validation sanity check",
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Training",
            initial=self._train_batch_index,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Validating",
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def on_sanity_check_start(self, num_sanity_check_steps: int):
        self.val_progress_bar = self.init_sanity_tqdm()
        self.val_progress_bar.total = num_sanity_check_steps
        self.main_progress_bar = tqdm(disable=True)  # dummy progress bar

    def on_sanity_check_end(self):
        self.main_progress_bar.close()
        self.val_progress_bar.close()

    def on_train_start(self):
        self.main_progress_bar = self.init_train_tqdm()

    def on_epoch_start(self, epoch: int):
        self._train_batch_index = 0
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset()
        self.main_progress_bar.set_description(f"Epoch {epoch}")

    def on_batch_end(self, metrics):
        train_monitor_metrics = self._get_train_monitor_metrics(metrics)
        self._train_batch_index += 1
        if self.is_enabled and self._train_batch_index % self.refresh_rate == 0:
            self.main_progress_bar.update(self.refresh_rate)
            self.main_progress_bar.set_postfix(**train_monitor_metrics, v=self.version)

    def on_epoch_end(self):
        print(" ")

    def on_validation_start(self, val_num_batches: int):
        self._val_batch_index = 0
        self.val_progress_bar = self.init_validation_tqdm()
        self.val_progress_bar.total = val_num_batches

    def on_validation_batch_end(self):
        self._val_batch_index += 1
        if self.is_enabled and self._val_batch_index % self.refresh_rate == 0:
            self.val_progress_bar.update(self.refresh_rate)
            self.main_progress_bar.update(self.refresh_rate)

    def on_validation_end(self, metrics):
        val_monitor_metrics = self._get_val_monitor_metrics(metrics)
        self.main_progress_bar.set_postfix(**val_monitor_metrics, v=self.version)
        self.val_progress_bar.close()

    def on_train_end(self):
        self.main_progress_bar.close()

    def _get_train_monitor_metrics(self, metrics):
        return {
            metric: score
            for metric, score in metrics.items()
            if metric in self.train_monitor_metrics
        }

    def _get_val_monitor_metrics(self, metrics):
        return {
            metric: score for metric, score in metrics.items() if metric in self.val_monitor_metrics
        }
