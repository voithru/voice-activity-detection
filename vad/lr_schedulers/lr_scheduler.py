from abc import ABC, abstractmethod


class LRScheduler(ABC):
    @abstractmethod
    def get_factor(self, step: int) -> float:
        pass


class ConstantLRScheduler(LRScheduler):
    def get_factor(self, step: int) -> float:
        return 1
