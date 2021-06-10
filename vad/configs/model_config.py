from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class SelfAttentionVADConfig:
    num_layers: int = MISSING
    d_model: int = MISSING
    dropout: float = MISSING


@dataclass
class DNNConfig:
    dropout: float = MISSING


@dataclass
class ACAMConfig:
    dropout: float = MISSING


@dataclass
class BoostedDNNConfig:
    dropout: float = MISSING


@dataclass
class ModelConfig:
    name: str = MISSING
    dnn: Optional[DNNConfig] = None
    boosted_dnn: Optional[BoostedDNNConfig] = None
    acam: Optional[ACAMConfig] = None
    self_attention: Optional[SelfAttentionVADConfig] = None
