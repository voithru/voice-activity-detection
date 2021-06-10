from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ContextResolutionConfig:
    context_window_half_frames: int = MISSING
    context_window_jump_frames: int = MISSING
    context_window_shift_frames: int = MISSING
