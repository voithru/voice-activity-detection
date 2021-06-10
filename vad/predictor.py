import math
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from more_itertools import ichunked
from omegaconf import OmegaConf
from scipy.special import softmax
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from vad.acoustics.feature_extractor import FeatureExtractor
from vad.configs.train_config import TrainConfig
from vad.data_models.audio_data import AudioData
from vad.data_models.voice_activity import Activity, VoiceActivity
from vad.models.model_factory import create_model
from vad.postprocessing.convert import convert_frames_to_samples, convert_samples_to_segments
from vad.postprocessing.split import optimal_split_voice_activity
from vad.postprocessing.trim import trim_voice_activity


@dataclass
class VADPredictParameters:
    split_max_seconds: float
    threshold: float
    min_vally_ms: int
    min_hill_ms: int
    hang_before_ms: int
    hang_over_ms: int
    activity_max_seconds: int
    return_probs: bool
    probs_sample_rate: Optional[int]
    show_progress_bar: bool


class VADFromScratchPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor: FeatureExtractor,
        device: torch.device,
        config: TrainConfig,
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.config: TrainConfig = config

        self.context_window_half_frames = config.context_resolution.context_window_half_frames
        self.context_window_jump_frames = config.context_resolution.context_window_jump_frames

        self.context_window_frames = (
            2 * (self.context_window_half_frames - 1) // self.context_window_jump_frames + 3
        )
        hop_samples = int(self.feature_extractor.config.transform.hop_ms / 1000 * 16000)
        if self.feature_extractor.config.transform is None:
            self.feature_window_half_size = self.context_window_half_frames * hop_samples
            self.feature_window_jump_size = 1
            self.feature_window_one_unit = hop_samples

        else:
            self.feature_window_half_size = self.context_window_half_frames
            self.feature_window_jump_size = self.context_window_jump_frames
            self.feature_window_one_unit = 1

    def predict_from_path(
        self, audio_path: Path, parameters: VADPredictParameters
    ) -> VoiceActivity:
        audio_data = AudioData.load(audio_path)
        return self.predict(audio_data, parameters)

    def predict(self, audio_data: AudioData, parameters: VADPredictParameters) -> VoiceActivity:
        if parameters.split_max_seconds is not None:
            num_chunks = math.ceil(
                audio_data.duration.total_seconds() / parameters.split_max_seconds
            )
        else:
            num_chunks = 1
        adjusted_chunk_seconds = audio_data.duration.total_seconds() / num_chunks
        voice_activity_chunks = []
        for chunk_index in tqdm(range(num_chunks), disable=not parameters.show_progress_bar):
            start_sample = int(chunk_index * adjusted_chunk_seconds * audio_data.sample_rate)
            end_sample = int((chunk_index + 1) * adjusted_chunk_seconds * audio_data.sample_rate)
            chunk_audio_data = AudioData(
                audio_data.audio[start_sample:end_sample],
                sample_rate=audio_data.sample_rate,
                duration=timedelta(seconds=adjusted_chunk_seconds),
            )
            frame_probabilities = self.predict_probabilities(chunk_audio_data)
            boosted_frame_probabilities = frame_probabilities.mean(axis=1)
            boosted_frame_predictions = boosted_frame_probabilities > parameters.threshold

            hop_ms = self.feature_extractor.config.transform.hop_ms
            window_ms = self.feature_extractor.config.transform.window_ms
            trimmed_frame_predictions = trim_voice_activity(
                boosted_frame_predictions,
                min_vally=round(parameters.min_vally_ms / hop_ms),
                min_hill=round(parameters.min_hill_ms / hop_ms),
                hang_before=round(parameters.hang_before_ms / hop_ms),
                hang_over=round(parameters.hang_over_ms / hop_ms),
            )

            sample_predictions = convert_frames_to_samples(
                trimmed_frame_predictions,
                sample_rate=16000,
                hop_ms=hop_ms,
                window_ms=window_ms,
            )

            if parameters.activity_max_seconds is not None and parameters.activity_max_seconds > 0:
                sample_full_probs = convert_frames_to_samples(
                    boosted_frame_probabilities,
                    sample_rate=16000,
                    hop_ms=hop_ms,
                    window_ms=window_ms,
                )

                sample_predictions = optimal_split_voice_activity(
                    sample_predictions=sample_predictions,
                    sample_probs=sample_full_probs,
                    max_length_seconds=parameters.activity_max_seconds,
                    sample_rate=16000,
                )

            segment_predictions = convert_samples_to_segments(sample_predictions, sample_rate=16000)

            activities = []
            for start_time, end_time in segment_predictions:
                activities.append(Activity(start=start_time, end=end_time))

            if parameters.return_probs:
                sample_probabilities = convert_frames_to_samples(
                    boosted_frame_probabilities,
                    sample_rate=parameters.probs_sample_rate,
                    hop_ms=hop_ms,
                    window_ms=window_ms,
                )
                probs = sample_probabilities.tolist()
            else:
                probs = None

            vad_prediction = VoiceActivity(
                duration=chunk_audio_data.duration,
                activities=activities,
                probs_sample_rate=parameters.probs_sample_rate if parameters.return_probs else None,
                probs=probs,
            )
            voice_activity_chunks.append(vad_prediction)

        merged_voice_activity = merge_voice_activities(voice_activity_chunks)

        return merged_voice_activity

    def predict_probabilities(self, audio_data: AudioData) -> np.array:
        feature = self.feature_extractor.extract_with_postprocessing(audio_data)

        if self.feature_extractor.config.transform is None:
            hop_samples = int(
                self.feature_extractor.config.transform.hop_ms / 1000 * audio_data.sample_rate
            )
            label_length = len(feature) // hop_samples
        else:
            label_length = len(feature)
        data_length = label_length - 2 * self.config.context_resolution.context_window_half_frames

        # if self.config.feature_extractor.normalization_path["--normalization-path"] is not None:
        #     global_normalization_factor_path = Path(self.config["--normalization-path"])
        #     with global_normalization_factor_path.open("rb") as global_normalization_factor_file:
        #         global_normalization_factor = pickle.load(global_normalization_factor_file)
        #         global_mean = global_normalization_factor["global_mean"]
        #         global_std = global_normalization_factor["global_std"]
        #
        #     feature = (feature - global_mean) / global_std

        chunk_size = 1000
        outputs = []
        for chunk in ichunked(range(data_length), chunk_size):
            inputs = []
            for item in chunk:
                feature_center = self.feature_window_half_size + item * self.feature_window_one_unit
                feature_left_neighbors = np.arange(
                    -self.feature_window_half_size, 0, self.feature_window_jump_size
                )
                feature_center_neighbors = np.array([0])
                feature_right_neighbors = np.arange(
                    1,
                    self.feature_window_half_size + self.feature_window_one_unit,
                    self.feature_window_jump_size,
                )

                feature_relative_neighbors = np.concatenate(
                    [feature_left_neighbors, feature_center_neighbors, feature_right_neighbors],
                    axis=0,
                )
                feature_neighbors = feature_center + feature_relative_neighbors

                feature_window = feature[feature_neighbors]

                label_center = self.context_window_half_frames + item
                label_left_neighbors = np.arange(
                    -self.context_window_half_frames, 0, self.context_window_jump_frames
                )
                label_center_neighbors = np.array([0])
                label_right_neighbors = np.arange(
                    1, self.context_window_half_frames + 1, self.context_window_jump_frames
                )

                label_relative_neighbors = np.concatenate(
                    [label_left_neighbors, label_center_neighbors, label_right_neighbors], axis=0
                )
                label_neighbors = label_center + label_relative_neighbors

                inputs.append({"feature": feature_window, "positions": label_neighbors})

            batch_inputs = default_collate(batch=inputs)
            self.model.eval()
            with torch.no_grad():
                batch_inputs["feature"] = batch_inputs["feature"].to(device=self.device)
                model_outputs = self.model(features=batch_inputs["feature"])
            probabilities = torch.nn.functional.softmax(model_outputs, dim=-1).view(-1, 2)[:, 1]

            output = {
                "probabilities": probabilities,
                "outputs": model_outputs,
                "positions": batch_inputs["positions"],
            }
            outputs.append(output)

        if self.config.model.name in ("dnn",):
            all_outputs = np.concatenate(outputs, axis=0)
            probabilities = softmax(all_outputs, axis=1)
            positive_probabilities: np.ndarray = probabilities[:, 1]
        elif self.config.model.name in ("bdnn", "acam", "self-attention"):
            boosted_outputs = np.zeros(
                shape=(label_length, self.context_window_frames, 2), dtype=np.float32
            )
            boosted_counts = np.zeros(
                shape=(label_length, self.context_window_frames, 1), dtype=np.float32
            )

            for output in outputs:
                outputs_array = output["outputs"].cpu().numpy()
                positions_array = output["positions"].cpu().numpy()

                window_indices = np.arange(self.context_window_frames)
                window_indices = np.expand_dims(window_indices, axis=0).repeat(
                    len(positions_array), axis=0
                )
                boosted_outputs[positions_array, window_indices] = outputs_array
                boosted_counts[positions_array, window_indices] = 1

            probabilities = softmax(boosted_outputs, axis=2)
            positive_probabilities = probabilities[:, :, 1]
        else:
            raise NotImplementedError

        return positive_probabilities

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config: TrainConfig = OmegaConf.create(checkpoint["config"])
        feature_extractor = FeatureExtractor(config.feature_extractor, use_spec_augment=False)

        context_window_frames = (
            2
            * (config.context_resolution.context_window_half_frames - 1)
            // config.context_resolution.context_window_jump_frames
            + 3
        )

        model = create_model(config.model, feature_extractor.feature_size, context_window_frames)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device=device)
        return cls(model=model, feature_extractor=feature_extractor, device=device, config=config)


def merge_voice_activities(voice_activities: List[VoiceActivity]) -> VoiceActivity:
    offset = timedelta(0)
    new_activities = []
    for voice_activity in voice_activities:
        for activity in voice_activity.activities:
            new_activity = Activity(start=activity.start + offset, end=activity.end + offset)
            new_activities.append(new_activity)
        offset += voice_activity.duration

    new_probs = None
    if voice_activities[0].probs:
        new_probs = list(chain(*[voice_activity.probs for voice_activity in voice_activities]))

    merged_voice_activity = VoiceActivity(
        duration=sum(
            [voice_activity.duration for voice_activity in voice_activities], timedelta(0)
        ),
        activities=new_activities,
        probs_sample_rate=voice_activities[0].probs_sample_rate,
        probs=new_probs,
    )
    return merged_voice_activity
