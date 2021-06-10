import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import yaml
from pytz import timezone

from vad.training.loggers.logger import Logger

KST = timezone("Asia/Seoul")
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S %Z"


class FileLogger(Logger):
    log_dir: Path
    logs_path: Path
    config_path: Path
    period: int

    def __init__(self, log_dir: Path, period=1):
        self.log_dir = log_dir
        self.logs_path = log_dir.joinpath("logs.json")
        self.config_path = log_dir.joinpath("config.yaml")
        self.period = period

    def log_metrics_at_intervals(self, metrics: dict) -> None:
        if not metrics["global_step"] % self.period == 0:
            return
        self.log_metrics(metrics)

    def log_metrics(self, metrics: dict) -> None:
        log = OrderedDict(
            **metrics,
            created_time=datetime.now(tz=KST).strftime(DATETIME_FORMAT),
        )
        log_line = json.dumps(log, ensure_ascii=False)
        with self.logs_path.open("a+") as logs_file:
            logs_file.write(log_line + "\n")

    def save_config(self, config: dict) -> None:
        with self.config_path.open("w") as config_file:
            yaml.dump(config, config_file)
