from statistics import harmonic_mean

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, roc_curve


def accuracy_metric_tensor(targets, predictions):
    correct = targets.view(-1) == predictions.view(-1)
    return torch.as_tensor(correct, dtype=torch.float).mean()


# Adapted from https://yangcha.github.io/EER-ROC/
def equal_error_rate(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1)
    # threshold = interp1d(fpr, thresholds)(eer)
    return eer


def vad_accuracy(frames_true, frames_pred, L=5):
    acc = accuracy_score(frames_true, frames_pred)  # frames accuracy

    start_boundaries, end_boundaries, num_segments_true = detect_boundaries(frames_true)
    _, _, num_segments_pred = detect_boundaries(frames_pred)

    sba = start_boundary_accuracy(
        frames_true,
        frames_pred,
        start_boundaries,
        num_segments=num_segments_true,
        L=L,
    )
    eba = end_boundary_accuracy(
        frames_true,
        frames_pred,
        end_boundaries,
        num_segments=num_segments_true,
        L=L,
    )

    bp = border_precision(
        sba,
        eba,
        num_segments_true=num_segments_true,
        num_segments_pred=num_segments_pred,
    )

    vacc = harmonic_mean([acc, sba, eba, bp])
    return vacc, acc, sba, eba, bp


def start_boundary_accuracy(frames_true, frames_pred, start_boundaries, num_segments, L):
    max_length = len(frames_true)
    sba_numerator = 0
    for start_boundary in start_boundaries:
        interval_start = max(start_boundary - L, 0)
        interval_end = min(start_boundary + L, max_length)
        sba_utterance_numerator = 0
        sba_utterance_denominator = 0
        for index in range(interval_start, interval_end):
            weight = weighting_function(index - start_boundary)
            delta = kronecker_delta(frames_pred[index], frames_true[index])
            sba_utterance_numerator += weight * delta
            sba_utterance_denominator += weight

        sba_numerator += sba_utterance_numerator / sba_utterance_denominator

    if num_segments > 0:
        sba = sba_numerator / num_segments
    else:
        sba = 0
    return sba


def end_boundary_accuracy(frames_true, frames_pred, end_boundaries, num_segments, L):
    max_length = len(frames_true)
    eba_numerator = 0
    for end_boundary in end_boundaries:
        interval_start = max(end_boundary - L, 0)
        interval_end = min(end_boundary + L, max_length)
        eba_utterance_numerator = 0
        eba_utterance_denominator = 0
        for index in range(interval_start, interval_end):
            weight = weighting_function(end_boundary - index)
            delta = kronecker_delta(frames_pred[index], frames_true[index])
            eba_utterance_numerator += weight * delta
            eba_utterance_denominator += weight

        eba_numerator += eba_utterance_numerator / eba_utterance_denominator

    if num_segments > 0:
        eba = eba_numerator / num_segments
    else:
        eba = 0
    return eba


def border_precision(sba, eba, num_segments_true, num_segments_pred):
    if num_segments_pred > 0:
        bp = num_segments_true / (2 * num_segments_pred) * (sba + eba)
    else:
        bp = 0
    return bp


def detect_boundaries(frames):
    # [1, 1, 1, 0, 0, 1, 1] -> [1, 0, 0, -1, 0, 1, 0, -1]
    boundaries = np.append(frames, np.array(0)) - np.append(np.array(0), frames)

    start_boundaries = np.where(boundaries == 1)[0]
    end_boundaries_plus_one = np.where(boundaries == -1)[0]
    end_boundaries = end_boundaries_plus_one - 1
    num_segments = len(start_boundaries)
    return start_boundaries, end_boundaries, num_segments


def weighting_function(x):
    if x >= 0:
        return 1
    else:
        return 0


def kronecker_delta(x, y):
    if x == y:
        return 1
    else:
        return 0
