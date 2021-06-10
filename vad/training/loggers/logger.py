from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log_metrics_at_intervals(self, metrics: dict) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict) -> None:
        pass

    @abstractmethod
    def save_config(self, config: dict) -> None:
        pass
