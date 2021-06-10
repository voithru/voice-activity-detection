import numpy as np


def split_voice_activity(segments, max_length_seconds=300):
    splitted_segments = []
    for start_time, end_time in segments:
        duration = end_time - start_time
        duration_seconds = duration.total_seconds()
        if duration_seconds > max_length_seconds:
            num_splits = int(duration_seconds // max_length_seconds)
            split_size = duration / num_splits

            for i in range(num_splits):
                split_start_time = start_time + i * split_size
                if i < num_splits - 1:
                    split_end_time = split_start_time + split_size
                else:  # last split
                    split_end_time = end_time
                splitted_segments.append((split_start_time, split_end_time))
        else:
            splitted_segments.append((start_time, end_time))

    return splitted_segments


def optimal_split_voice_activity(
    sample_predictions, sample_probs, max_length_seconds=300, sample_rate=16000
):
    max_samples = max_length_seconds * sample_rate
    split_predictions = sample_predictions.copy()

    is_voice = False
    switch_to_speech = False
    switch_to_non_speech = False
    sample_index = None
    start_index = None
    end_index = None

    for sample_index, (sample_prediction, sample_prob) in enumerate(
        zip(sample_predictions, sample_probs)
    ):
        if sample_prediction == 1 and not is_voice:
            switch_to_speech = True
            is_voice = True
        if sample_prediction == 0 and is_voice:
            switch_to_non_speech = True
            is_voice = False

        if switch_to_speech:
            start_index = sample_index
            switch_to_speech = False
        if switch_to_non_speech:
            end_index = sample_index
            switch_to_non_speech = False

        if start_index is not None and end_index is not None:
            if end_index - start_index > max_samples:
                break_points = optimal_split_long_block(
                    block_sample_probs=sample_probs[start_index:end_index], max_samples=max_samples
                )
                for break_point in break_points:
                    split_predictions[start_index + break_point] = 0
            start_index = None
            end_index = None
    if sample_index is not None and start_index is not None and is_voice:
        end_index = sample_index + 1
        if end_index - start_index > max_samples:
            break_points = optimal_split_long_block(
                block_sample_probs=sample_probs[start_index:end_index], max_samples=max_samples
            )
            for break_point in break_points:
                split_predictions[start_index + break_point] = 0

    return split_predictions


def optimal_split_long_block(block_sample_probs, max_samples):
    assert max_samples > 1
    half_max_samples = max_samples // 2
    trimmed_block = block_sample_probs[half_max_samples:-half_max_samples]
    break_point = half_max_samples + np.argmin(trimmed_block)

    left_block = block_sample_probs[:break_point]
    right_block = block_sample_probs[break_point + 1 :]

    if len(left_block) > max_samples:
        left_break_points = optimal_split_long_block(
            block_sample_probs=left_block, max_samples=max_samples
        )
    else:
        left_break_points = []
    if len(right_block) > max_samples:
        right_break_points_from_zero = optimal_split_long_block(
            block_sample_probs=right_block, max_samples=max_samples
        )
        right_break_points = [
            break_point + 1 + right_break_point
            for right_break_point in right_break_points_from_zero
        ]
    else:
        right_break_points = []

    break_points = left_break_points + [break_point] + right_break_points
    return break_points
