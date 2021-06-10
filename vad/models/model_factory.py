from enum import Enum

from vad.configs.model_config import ModelConfig

from .acam import ACAM
from .boosted_dnn import BoostedDNN
from .dnn import DNN
from .self_attention import SelfAttentiveVAD


class ModelName(Enum):
    DNN = "dnn"
    BDNN = "bdnn"
    ACAM = "acam"
    SELF_ATTENTIVE = "self-attention"


def create_model(model_config: ModelConfig, feature_size: int, context_window_frames: int):
    name = ModelName(model_config.name)
    window_feature_size = feature_size * context_window_frames

    if name == ModelName.DNN:
        classifier = DNN(window_feature_size, 512, 512, model_config.dnn.dropout)
    elif name == ModelName.BDNN:
        classifier = BoostedDNN(
            window_feature_size,
            context_window_frames,
            512,
            512,
            model_config.boosted_dnn.dropout,
        )
    elif name == ModelName.ACAM:
        classifier = ACAM(
            window_feature_size,
            context_window_frames,
            128,
            128,
            128,
            model_config.acam.dropout,
            7,
        )
    elif name == ModelName.SELF_ATTENTIVE:
        classifier = SelfAttentiveVAD(
            feature_size,
            model_config.self_attention.num_layers,
            model_config.self_attention.d_model,
            model_config.self_attention.dropout,
        )
    else:
        raise NotImplementedError

    model = classifier

    return model
