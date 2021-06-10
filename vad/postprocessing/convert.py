from datetime import timedelta

import numpy as np


def convert_frames_to_samples(frames, sample_rate=16000, hop_ms=10, window_ms=10):

    hop_samples = sample_rate * hop_ms / 1000
    window_samples = sample_rate * window_ms / 1000

    num_samples = int((len(frames) - 1) * hop_samples + window_samples)
    samples = np.zeros(shape=num_samples)
    counts = np.zeros(shape=num_samples)

    start_index = 0
    for frame_label in frames:
        samples[int(start_index) : int(start_index + window_samples)] += frame_label
        counts[int(start_index) : int(start_index + window_samples)] += 1

        start_index += hop_samples

    counts = (counts == 0).choose(counts, 1)
    samples_averaged = samples / counts
    return samples_averaged


def convert_samples_to_segments(samples, sample_rate=16000):

    segments = []

    is_voice = False
    switch_to_speech = False
    switch_to_non_speech = False
    sample_index = None
    start_time = None
    end_time = None

    for sample_index, value in enumerate(samples):
        if value == 1 and not is_voice:
            switch_to_speech = True
            is_voice = True
        if value == 0 and is_voice:
            switch_to_non_speech = True
            is_voice = False

        if switch_to_speech:
            start_time = timedelta(seconds=sample_index / sample_rate)
            switch_to_speech = False
        if switch_to_non_speech:
            end_time = timedelta(seconds=(sample_index - 1) / sample_rate)
            switch_to_non_speech = False

        if start_time is not None and end_time is not None:
            segments.append((start_time, end_time))
            start_time = None
            end_time = None
    if sample_index is not None and is_voice:
        end_time = timedelta(seconds=sample_index / sample_rate)
        segments.append((start_time, end_time))

    return segments
