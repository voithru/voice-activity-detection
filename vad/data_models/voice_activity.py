import json
import math
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np

from vad.util.time_utils import (
    format_timedelta_to_milliseconds,
    format_timedelta_to_timecode,
    parse_timecode_to_timedelta,
)


class VoiceActivityVersion(Enum):
    v01 = "v0.1"
    v02 = "v0.2"
    v03 = "v0.3"


class VoiceActivityMillisecondsVersion(Enum):
    v01 = "v0.1"
    v02 = "v0.2"
    v03 = "v0.3"


@dataclass
class Activity:
    start: timedelta
    end: timedelta


@dataclass
class VoiceActivity:
    duration: timedelta
    activities: List[Activity]
    probs_sample_rate: Optional[int]
    probs: Optional[List[float]]

    @classmethod
    def load(cls, path: Path):
        with path.open() as file:
            voice_activity_data = json.load(file)
        return VoiceActivity.from_json(voice_activity_data)

    @classmethod
    def from_json(cls, voice_activity_data: dict):
        version = voice_activity_data["version"]
        if version == VoiceActivityVersion.v01.value:
            voice_activity = cls(
                duration=parse_timecode_to_timedelta(voice_activity_data["duration"]),
                activities=[
                    Activity(
                        start=parse_timecode_to_timedelta(speech_block["start_time"]),
                        end=parse_timecode_to_timedelta(speech_block["end_time"]),
                    )
                    for speech_block in voice_activity_data["voice_activity"]
                ],
                probs_sample_rate=voice_activity_data.get("probs_sample_rate"),
                probs=voice_activity_data.get("probs"),
            )
        elif version == VoiceActivityVersion.v02.value:
            if voice_activity_data["time_format"] == "timecode":
                voice_activity = cls(
                    duration=parse_timecode_to_timedelta(voice_activity_data["duration"]),
                    activities=[
                        Activity(
                            start=parse_timecode_to_timedelta(speech_block["start_time"]),
                            end=parse_timecode_to_timedelta(speech_block["end_time"]),
                        )
                        for speech_block in voice_activity_data["voice_activity"]
                    ],
                    probs_sample_rate=voice_activity_data.get("probs_sample_rate"),
                    probs=voice_activity_data.get("probs"),
                )
            elif voice_activity_data["time_format"] == "millisecond":
                voice_activity = cls(
                    duration=timedelta(milliseconds=voice_activity_data["duration"]),
                    activities=[
                        Activity(
                            start=timedelta(milliseconds=speech_block["start_time"]),
                            end=timedelta(milliseconds=speech_block["end_time"]),
                        )
                        for speech_block in voice_activity_data["voice_activity"]
                    ],
                    probs_sample_rate=voice_activity_data.get("probs_sample_rate"),
                    probs=voice_activity_data.get("probs"),
                )
            else:
                raise NotImplementedError
        elif version == VoiceActivityVersion.v03.value:
            voice_activity = cls(
                duration=parse_timecode_to_timedelta(voice_activity_data["duration"]),
                activities=[
                    Activity(
                        start=parse_timecode_to_timedelta(activity["start"]),
                        end=parse_timecode_to_timedelta(activity["end"]),
                    )
                    for activity in voice_activity_data["activities"]
                ],
                probs_sample_rate=voice_activity_data.get("probs_sample_rate"),
                probs=voice_activity_data.get("probs"),
            )
        else:
            raise NotImplementedError
        return voice_activity

    def save(self, path: Path, version: VoiceActivityVersion = VoiceActivityVersion.v03):
        voice_activity_data = self.to_json(version)
        with path.open("w") as file:
            json.dump(voice_activity_data, file, ensure_ascii=False, indent=4)

    def to_json(self, version: VoiceActivityVersion = VoiceActivityVersion.v03):
        if version == VoiceActivityVersion.v01:
            voice_activity_formatted = {
                "version": VoiceActivityVersion.v01.value,
                "duration": format_timedelta_to_timecode(self.duration),
                "voice_activity": [
                    {
                        "start_time": format_timedelta_to_timecode(activity.start),
                        "end_time": format_timedelta_to_timecode(activity.end),
                    }
                    for activity in self.activities
                ],
                "probs_sample_rate": self.probs_sample_rate,
                "probs": self.probs,
            }
        elif version == VoiceActivityVersion.v02:
            voice_activity_formatted = {
                "version": VoiceActivityVersion.v02.value,
                "duration": format_timedelta_to_timecode(self.duration),
                "time_format": "timecode",
                "voice_activity": [
                    {
                        "start_time": format_timedelta_to_timecode(activity.start),
                        "end_time": format_timedelta_to_timecode(activity.end),
                    }
                    for activity in self.activities
                ],
                "probs_sample_rate": self.probs_sample_rate,
                "probs": self.probs,
            }
        elif version == VoiceActivityVersion.v03:
            voice_activity_formatted = {
                "version": VoiceActivityVersion.v03.value,
                "duration": format_timedelta_to_timecode(self.duration),
                "activities": [
                    {
                        "start": format_timedelta_to_timecode(activity.start),
                        "end": format_timedelta_to_timecode(activity.end),
                    }
                    for activity in self.activities
                ],
                "probs_sample_rate": self.probs_sample_rate,
                "probs": self.probs,
            }
        else:
            raise NotImplementedError
        return voice_activity_formatted

    def to_milliseconds(
        self, version: VoiceActivityMillisecondsVersion = VoiceActivityMillisecondsVersion.v03
    ):
        if version == VoiceActivityMillisecondsVersion.v02:
            voice_activity_milliseconds = {
                "version": version.value,
                "duration": format_timedelta_to_milliseconds(self.duration),
                "time_format": "millisecond",
                "voice_activity": [
                    {
                        "start_time": format_timedelta_to_milliseconds(activity.start),
                        "end_time": format_timedelta_to_milliseconds(activity.end),
                    }
                    for activity in self.activities
                ],
                "probs_sample_rate": self.probs_sample_rate,
                "probs": self.probs,
            }
        elif version == VoiceActivityMillisecondsVersion.v03:
            voice_activity_milliseconds = {
                "version": version.value,
                "duration": {"total_milliseconds": format_timedelta_to_milliseconds(self.duration)},
                "activities": [
                    {
                        "start": {
                            "total_milliseconds": format_timedelta_to_milliseconds(activity.start)
                        },
                        "end": {
                            "total_milliseconds": format_timedelta_to_milliseconds(activity.end)
                        },
                    }
                    for activity in self.activities
                ],
                "probs_sample_rate": self.probs_sample_rate,
                "probs": self.probs,
            }
        else:
            raise NotImplementedError
        return voice_activity_milliseconds

    @classmethod
    def from_milliseconds(cls, voice_activity_data: dict):
        version = voice_activity_data["version"]  # version of milliseconds format
        if version == VoiceActivityMillisecondsVersion.v02.value:
            voice_activity = VoiceActivity(
                duration=timedelta(milliseconds=voice_activity_data["duration"]),
                activities=[
                    Activity(
                        start=timedelta(milliseconds=speech_block["start_time"]),
                        end=timedelta(milliseconds=speech_block["end_time"]),
                    )
                    for speech_block in voice_activity_data["voice_activity"]
                ],
                probs_sample_rate=voice_activity_data.get("probs_sample_rate"),
                probs=voice_activity_data.get("probs"),
            )
        elif version == VoiceActivityMillisecondsVersion.v03.value:
            voice_activity = VoiceActivity(
                duration=timedelta(
                    milliseconds=voice_activity_data["duration"]["total_milliseconds"]
                ),
                activities=[
                    Activity(
                        start=timedelta(milliseconds=segment["start"]["total_milliseconds"]),
                        end=timedelta(milliseconds=segment["end"]["total_milliseconds"]),
                    )
                    for segment in voice_activity_data["activities"]
                ],
                probs_sample_rate=voice_activity_data.get("probs_sample_rate"),
                probs=voice_activity_data.get("probs"),
            )
        else:
            raise NotImplementedError
        return voice_activity

    def to_labels(self, sample_rate: int) -> np.array:
        total_samples = int(self.duration.total_seconds() * sample_rate)
        labels = np.zeros(total_samples, dtype=np.long)
        for activity in self.activities:
            start_sample = int(activity.start.total_seconds() * sample_rate)
            end_sample = int(activity.end.total_seconds() * sample_rate)
            labels[start_sample:end_sample] = 1
        return labels
