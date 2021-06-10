import multiprocessing
import pickle
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import numpy as np
import soundfile
from scipy.io import loadmat
from torch.utils.data import Dataset
from tqdm import tqdm

from vad.acoustics.speech_noise_mix import mix_speech_noise
from vad.data_models.audio_data import AudioData
from vad.data_models.vad_data import VADDataPair
from vad.data_models.voice_activity import VoiceActivity


class ResolutionMapDataset(Dataset):
    variable_length_fields = {}

    def __init__(
        self,
        data_pairs: List[VADDataPair],
        data_dir: Path,
        noise_paths: Optional[List[Path]],
        noise_ratio,
        min_snr,
        max_snr,
        feature_extractor,
        context_window_half_frames: int,
        context_window_jump_frames: int,
        context_window_shift_frames: int = 1,
        expand_target=True,
        global_normalization_factor_path: Path = None,
        calculate_global_normalization_factor: bool = False,
        num_workers: int = 0,
    ):
        with TemporaryDirectory() as mix_dir:
            mix_dir = Path(mix_dir)

            target_data_paths = []
            for data_pair in data_pairs:
                audio_path = data_dir.joinpath(data_pair.audio_path)
                audio_data = AudioData.load(audio_path)
                voice_activity_path = data_dir.joinpath(data_pair.voice_activity_path)
                if noise_paths is not None:
                    noisy_speech = mix_speech_noise(
                        speech_path=audio_path,
                        noise_paths=noise_paths,
                        noise_ratio=noise_ratio,
                        min_snr=min_snr,
                        max_snr=max_snr,
                    )
                    noisy_speech_path = mix_dir.joinpath(audio_path.name)
                    soundfile.write(
                        file=noisy_speech_path,
                        data=noisy_speech,
                        samplerate=audio_data.sample_rate,
                    )
                    target_audio_path = noisy_speech_path
                else:
                    target_audio_path = audio_path

                target_data_paths.append((target_audio_path, voice_activity_path))

            if num_workers == 0:
                self.data = self.extract_data(
                    data_paths=target_data_paths,
                    feature_extractor=feature_extractor,
                )
            else:
                self.data = self.extract_data_multiprocessing(
                    data_paths=target_data_paths,
                    feature_extractor=feature_extractor,
                    num_workers=num_workers,
                )

        self.data_lengths = [
            (len(label) - 2 * context_window_half_frames - 1) // context_window_shift_frames + 1
            for feature, label in self.data
        ]
        print("lengths", [len(label) for feature, label in self.data])
        print("lengths", self.data_lengths)

        if global_normalization_factor_path is not None:
            if calculate_global_normalization_factor:
                global_mean = np.mean(
                    [feature.mean(axis=0) for feature, label in self.data], axis=0
                )
                global_std = np.mean([feature.std(axis=0) for feature, label in self.data], axis=0)
                global_normalization_factor = {"global_mean": global_mean, "global_std": global_std}
                global_normalization_factor_path.parent.mkdir(parents=True, exist_ok=True)
                with global_normalization_factor_path.open(
                    "wb"
                ) as global_normalization_factor_file:
                    pickle.dump(global_normalization_factor, global_normalization_factor_file)
            else:
                with global_normalization_factor_path.open(
                    "rb"
                ) as global_normalization_factor_file:
                    global_normalization_factor = pickle.load(global_normalization_factor_file)
                    global_mean = global_normalization_factor["global_mean"]
                    global_std = global_normalization_factor["global_std"]

            normalized_data = []
            for feature, label in self.data:
                feature = (feature - global_mean) / global_std
                normalized_data.append((feature, label))
            self.data = normalized_data

        self.context_window_half_frames = context_window_half_frames
        self.context_window_jump_frames = context_window_jump_frames
        self.context_window_shift_frames = context_window_shift_frames
        self.expand_target = expand_target

    def __getitem__(self, item):
        i = 0
        for i, length in enumerate(self.data_lengths):
            if item < length:
                break
            item = item - length

        feature, label = self.data[i]
        center = self.context_window_half_frames + item * self.context_window_shift_frames

        left_neighbors = np.arange(
            -self.context_window_half_frames, 0, self.context_window_jump_frames
        )
        center_neighbors = np.array([0])
        right_neighbors = np.arange(
            1, self.context_window_half_frames + 1, self.context_window_jump_frames
        )

        relative_neighbors = np.concatenate(
            [left_neighbors, center_neighbors, right_neighbors], axis=0
        )
        neighbors = center + relative_neighbors

        if self.expand_target:
            label_window = label[neighbors]
        else:
            label_window = label[center : center + 1]

        inputs = {
            "feature": feature[neighbors],
            "positions": neighbors,
            "data-index": i,
            "data-length": self.data_lengths[i],
        }
        targets = label_window

        return inputs, targets

    def __len__(self):
        return sum(self.data_lengths)

    @staticmethod
    def extract_data(data_paths, feature_extractor):
        data = []
        for audio_path, voice_activity_path in tqdm(
            data_paths, desc="Preprocessing data", leave=False
        ):
            feature, label = ResolutionMapDataset.extract_single_data(
                audio_path=audio_path,
                voice_activity_path=voice_activity_path,
                feature_extractor=feature_extractor,
            )
            data.append((feature, label))
        return data

    @staticmethod
    def extract_data_multiprocessing(data_paths, feature_extractor, num_workers):
        pbar = tqdm(total=len(data_paths), desc="Preprocessing data", leave=False)

        def pbar_update(*args):
            pbar.update()

        data_results = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            for audio_path, voice_activity_path in data_paths:
                data_result = pool.apply_async(
                    ResolutionMapDataset.extract_single_data,
                    kwds={
                        "audio_path": audio_path,
                        "voice_activity_path": voice_activity_path,
                        "feature_extractor": feature_extractor,
                    },
                    callback=pbar_update,
                )
                data_results.append(data_result)

            data = [data_result.get() for data_result in data_results]
        pbar.close()
        return data

    @staticmethod
    def extract_single_data(audio_path, voice_activity_path, feature_extractor):
        print("audio_path", audio_path)
        feature = feature_extractor.extract_from_path_with_postprocessing(audio_path=audio_path)

        if voice_activity_path.suffix == ".json":
            audio_data = AudioData.load(audio_path)
            voice_activity = VoiceActivity.load(voice_activity_path)
            hop_samples = int(feature_extractor.transform.hop_ms / 1000 * audio_data.sample_rate)
            labels = voice_activity.to_labels(sample_rate=audio_data.sample_rate // hop_samples)
        else:
            if voice_activity_path.suffix == ".npy":
                raw_label = np.load(voice_activity_path)
                raw_label = raw_label.astype(np.long)

            elif voice_activity_path.suffix == ".mat":
                loaded_mat = loadmat(str(voice_activity_path))
                raw_label = loaded_mat["y_label"].squeeze(axis=1).astype(np.long)
            else:
                raise NotImplementedError

            label_indices = np.arange(0, len(raw_label), step=feature_extractor.hop_samples)
            labels = raw_label[label_indices]

        return feature, labels


def parse_timecode(timecode):
    epoch = datetime(year=1900, month=1, day=1)
    return datetime.strptime(timecode, "%H:%M:%S.%f") - epoch
