from abc import ABC
from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class Feeder(ABC):
    train_dataloader: DataLoader
    val_dataloader: DataLoader
