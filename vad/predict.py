from pathlib import Path
from typing import Optional

import torch
from typer import Option

from vad.predictor import VADFromScratchPredictor, VADPredictParameters


def predict_vad_from_scratch(
    audio_path: Path,
    checkpoint_path: Path,
    output_path: Optional[Path] = Option(None, help="Path to store output. Default to stdout."),
    split_max_seconds: Optional[float] = Option(None, help="Chunk size to split audio in seconds."),
    activity_max_sec: Optional[int] = Option(
        None, help="Maximum length of voice activity in seconds"
    ),
    threshold: float = 0.5,
    min_vally_ms: int = 0,
    min_hill_ms: int = 0,
    hang_before_ms: int = 0,
    hang_over_ms: int = 0,
    return_probs: bool = False,
    probs_sample_rate: Optional[int] = None,
):
    predictor = VADFromScratchPredictor.from_checkpoint(
        checkpoint_path,
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
    voice_activity = predictor.predict_from_path(
        audio_path,
        VADPredictParameters(
            split_max_seconds,
            threshold,
            min_vally_ms,
            min_hill_ms,
            hang_before_ms,
            hang_over_ms,
            activity_max_sec,
            return_probs,
            probs_sample_rate,
            True,
        ),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice_activity.save(path=output_path)
    else:
        print(voice_activity)
