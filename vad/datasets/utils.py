import random
from pathlib import Path
from typing import Optional


def load_data_paths(data_list_path: Path, data_dir: Optional[Path]):
    if data_dir is None:
        data_dir = data_list_path.parent
    data_paths = []
    with data_list_path.open() as data_list:
        for line in data_list:
            audio_path, voice_activity_path = line.strip().split(",")
            if data_list_path is not None:
                audio_path = data_dir.joinpath(audio_path)
                voice_activity_path = data_dir.joinpath(voice_activity_path)
            else:
                audio_path = Path(audio_path)
                voice_activity_path = Path(voice_activity_path)
            data_paths.append((audio_path, voice_activity_path))

    random.shuffle(data_paths)
    return data_paths


def load_noise_paths(noise_list_path: Path, noise_data_dir: Optional[Path]):
    if noise_data_dir is None:
        noise_data_dir = noise_list_path.parent
    noise_paths = []
    with noise_list_path.open() as noise_list:
        for noise_path in noise_list:
            if noise_data_dir is not None:
                noise_path = noise_data_dir.joinpath(noise_path.strip())
            else:
                noise_path = Path(noise_path.strip())
            noise_paths.append(noise_path)
    return noise_paths
