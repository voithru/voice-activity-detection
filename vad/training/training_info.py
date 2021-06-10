from dataclasses import dataclass


@dataclass
class TrainingInfo:
    epoch: int
    global_step: int
    batch_index: int
