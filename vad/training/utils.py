import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def find_next_version_dir(runs_dir: Path, run_name: str) -> Path:
    run_dir = runs_dir.joinpath(run_name)
    version_prefix = "v"
    if not run_dir.exists():
        next_version = 0
    else:
        existing_versions = []
        for child_path in run_dir.iterdir():
            if child_path.is_dir() and child_path.name.startswith(version_prefix):
                existing_versions.append(int(child_path.name[len(version_prefix) :]))

        if len(existing_versions) == 0:
            last_version = -1
        else:
            last_version = max(existing_versions)

        next_version = last_version + 1
    version_dir = run_dir.joinpath(f"{version_prefix}{next_version:0>3}")
    version_dir.mkdir(parents=True)
    return version_dir


def seed(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def to_device(batch: Any, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device)
    elif isinstance(batch, Mapping):
        return {key: to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, Iterable):
        return [to_device(item, device) for item in batch]
    else:
        raise NotImplementedError


def to_float(batch: Any):
    if isinstance(batch, float):
        return batch
    elif isinstance(batch, torch.Tensor) or isinstance(batch, np.float):
        return float(batch)
    elif isinstance(batch, Mapping):
        return {key: to_float(value) for key, value in batch.items()}
    elif isinstance(batch, Iterable):
        return [to_float(item) for item in batch]
    else:
        return float(batch)


def to_numpy(batch: Any):
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu().numpy()
    elif isinstance(batch, Mapping):
        return {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    elif isinstance(batch, Iterable):
        return [
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
            for item in batch
        ]

    else:
        raise NotImplementedError


def dictionarize_list(results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    example_result = results[0]
    result_dict = {}
    for key in example_result.keys():
        result_dict[key] = [result[key] for result in results]
    return result_dict
