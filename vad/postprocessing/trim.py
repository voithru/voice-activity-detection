from itertools import chain, tee


def trim_voice_activity(predictions, min_vally=20, min_hill=20, hang_before=10, hang_over=10):

    predictions_copy = predictions.copy()

    onset = False
    onset_point = None
    offset = False
    offset_point = None

    # Fill valley
    if min_vally > 0:
        for index, (current_value, next_value) in enumerate(
            current_next_iterator(predictions_copy.tolist())
        ):
            next_index = index

            if current_value == 0 and next_value == 1:
                if offset:
                    if index - offset_point < min_vally:
                        predictions_copy[offset_point:next_index] = 1
                    offset = False

            elif current_value == 1 and next_value == 0:
                offset = True
                offset_point = next_index

    # Flatten hill
    if min_hill > 0:
        for index, (current_value, next_value) in enumerate(
            current_next_iterator(predictions_copy.tolist())
        ):
            next_index = index

            if current_value == 0 and next_value == 1:
                onset = True
                onset_point = next_index

            elif current_value == 1 and next_value == 0:
                if onset:
                    if index - onset_point < min_hill:
                        predictions_copy[onset_point:next_index] = 0
                    onset = False

    # Extend both ends
    if hang_before > 0 or hang_before > 0:
        for index, (current_value, next_value) in enumerate(
            current_next_iterator(predictions_copy.tolist())
        ):
            next_index = index

            if current_value == 0 and next_value == 1:
                if index < hang_before:
                    predictions_copy[0:next_index] = 1
                else:
                    predictions_copy[index - hang_before : next_index] = 1

            elif current_value == 1 and next_value == 0:
                if len(predictions) - hang_over < index:
                    predictions_copy[next_index:] = 1
                else:
                    predictions_copy[next_index : next_index + hang_over] = 1

    return predictions_copy


def current_next_iterator(iterable):
    currents, nexts = tee(iterable, 2)
    currents = chain([None], currents)
    return zip(currents, nexts)
