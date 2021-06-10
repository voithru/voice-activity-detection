import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from typer import Option

from vad.data_models.audio_data import AudioData
from vad.data_models.vad_data import VADDataList
from vad.data_models.voice_activity import VoiceActivity
from vad.metrics import equal_error_rate, vad_accuracy
from vad.predictor import VADFromScratchPredictor


def evaluate_vad_from_scratch(
    eval_path: Path,
    checkpoint_path: Path,
    output_path: Optional[Path] = Option(None, help="Path to store output. Default to stdout."),
    data_dir: Optional[Path] = None,
    threshold: float = 0.5,
    shuffle: bool = False,
    limit: Optional[int] = None,
    random_seed: int = 0,
):
    predictor: VADFromScratchPredictor = VADFromScratchPredictor.from_checkpoint(
        checkpoint_path,
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )

    if data_dir is None:
        data_dir = eval_path.parent

    eval_data_list: VADDataList = VADDataList.load(eval_path)

    eval_data_pairs = eval_data_list.pairs
    if shuffle:
        random.seed(random_seed)
        random.shuffle(eval_data_pairs)
    if limit:
        eval_data_pairs = eval_data_pairs[:limit]

    results = []
    for data_pair in tqdm(eval_data_pairs):
        audio_path = data_dir.joinpath(data_pair.audio_path)
        voice_activity_path = data_dir.joinpath(data_pair.voice_activity_path)

        true_voice_activity = VoiceActivity.load(voice_activity_path)
        true_labels = true_voice_activity.to_labels(100)

        audio_data = AudioData.load(audio_path)
        all_frame_probabilities = predictor.predict_probabilities(audio_data)
        middle_index = int(all_frame_probabilities.shape[1] / 2)
        single_frame_probabilities = all_frame_probabilities[:, middle_index]
        single_frame_probabilities = single_frame_probabilities[: len(true_labels)]
        single_frame_predictions = single_frame_probabilities > threshold
        boosted_frame_probabilities = all_frame_probabilities.mean(axis=1)
        boosted_frame_probabilities = boosted_frame_probabilities[: len(true_labels)]
        boosted_frame_predictions = boosted_frame_probabilities > threshold

        auc = roc_auc_score(true_labels, boosted_frame_probabilities)
        accuracy = accuracy_score(true_labels, boosted_frame_predictions)
        precision = precision_score(true_labels, boosted_frame_predictions)
        recall = recall_score(true_labels, boosted_frame_predictions)
        vacc, acc, sba, eba, bp = vad_accuracy(true_labels, single_frame_predictions)
        eer = equal_error_rate(true_labels, single_frame_predictions)

        boosted_auc = roc_auc_score(true_labels, boosted_frame_probabilities)
        boosted_accuracy = accuracy_score(true_labels, boosted_frame_predictions)
        boosted_precision = precision_score(true_labels, boosted_frame_predictions)
        boosted_recall = recall_score(true_labels, boosted_frame_predictions)

        boosted_vacc, boosted_acc, boosted_sba, boosted_eba, boosted_bp = vad_accuracy(
            true_labels, boosted_frame_predictions
        )
        boosted_eer = equal_error_rate(true_labels, boosted_frame_predictions)

        print(
            f"""
{data_pair.audio_path}
AUC: {auc:0.2%}
Accuracy: {accuracy:0.2%}
Precision: {precision:0.2%}
Recall: {recall:0.2%}
VACC: {vacc:0.2%}
SBA: {sba:0.2%}
EBA: {eba:0.2%}
BP: {bp:0.2%}
EER: {eer:0.2%}
Boosted AUC: {boosted_auc:0.2%}
Boosted Accuracy: {boosted_accuracy:0.2%}
Boosted Precision: {boosted_precision:0.2%}
Boosted Recall: {boosted_recall:0.2%}
Boosted VACC: {boosted_vacc:0.2%}
Boosted SBA: {boosted_sba:0.2%}
Boosted EBA: {boosted_eba:0.2%}
Boosted BP: {boosted_bp:0.2%}
Boosted EER: {boosted_eer:0.2%}
"""
        )

        result = OrderedDict(
            {
                "audio_path": str(audio_path),
                "voice_activity_path": str(voice_activity_path),
                "auc": auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "vacc": vacc,
                "sba": sba,
                "eba": eba,
                "bp": bp,
                "eer": eer,
                "boosted_auc": boosted_auc,
                "boosted_accuracy": boosted_accuracy,
                "boosted_precision": boosted_precision,
                "boosted_recall": boosted_recall,
                "boosted_vacc": boosted_vacc,
                "boosted_sba": boosted_sba,
                "boosted_eba": boosted_eba,
                "boosted_bp": boosted_bp,
                "boosted_eer": boosted_eer,
            }
        )

        results.append(result)

    total_result = {
        "auc": np.mean([result["auc"] for result in results]),
        "accuracy": np.mean([result["accuracy"] for result in results]),
        "precision": np.mean([result["precision"] for result in results]),
        "recall": np.mean([result["recall"] for result in results]),
        "vacc": np.mean([result["vacc"] for result in results]),
        "sba": np.mean([result["sba"] for result in results]),
        "eba": np.mean([result["eba"] for result in results]),
        "bp": np.mean([result["bp"] for result in results]),
        "eer": np.mean([result["eer"] for result in results]),
        "boosted_auc": np.mean([result["boosted_auc"] for result in results]),
        "boosted_accuracy": np.mean([result["boosted_accuracy"] for result in results]),
        "boosted_precision": np.mean([result["boosted_precision"] for result in results]),
        "boosted_recall": np.mean([result["boosted_recall"] for result in results]),
        "boosted_vacc": np.mean([result["boosted_vacc"] for result in results]),
        "boosted_sba": np.mean([result["boosted_sba"] for result in results]),
        "boosted_eba": np.mean([result["boosted_eba"] for result in results]),
        "boosted_bp": np.mean([result["boosted_bp"] for result in results]),
        "boosted_eer": np.mean([result["boosted_eer"] for result in results]),
    }

    print(
        f"""
Total:
AUC: {total_result['auc']:0.2%}
Accuracy: {total_result['accuracy']:0.2%}
Precision: {total_result['precision']:0.2%}
Recall: {total_result['recall']:0.2%}
VACC: {total_result['vacc']:0.2%}
SBA: {total_result['sba']:0.2%}
EBA: {total_result['eba']:0.2%}
BP: {total_result['bp']:0.2%}
EER: {total_result['eer']:0.2%}
Boosted AUC: {total_result['boosted_auc']:0.2%}
Boosted Accuracy: {total_result['boosted_accuracy']:0.2%}
Boosted Precision: {total_result['boosted_precision']:0.2%}
Boosted Recall: {total_result['boosted_recall']:0.2%}
Boosted VACC: {total_result['boosted_vacc']:0.2%}
Boosted SBA: {total_result['boosted_sba']:0.2%}
Boosted EBA: {total_result['boosted_eba']:0.2%}
Boosted BP: {total_result['boosted_bp']:0.2%}
Boosted EER: {total_result['boosted_eer']:0.2%}
"""
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as output_file:
            total_result_json = json.dumps(total_result, ensure_ascii=False)
            output_file.write(f"{total_result_json}\n")
            for result in results:
                result_json = json.dumps(result, ensure_ascii=False)
                output_file.write(f"{result_json}\n")
