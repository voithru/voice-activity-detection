import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class VADDataPair:
    audio_path: Path
    voice_activity_path: Path

    def to_json(self):
        return {
            "audio_path": str(self.audio_path),
            "voice_activity_path": str(self.voice_activity_path),
        }

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(
            audio_path=Path(json_data["audio_path"]),
            voice_activity_path=Path(json_data["voice_activity_path"]),
        )


@dataclass
class VADDataList:
    pairs: List[VADDataPair]

    def save(self, path: Path):
        with path.open("w") as file:
            for data_pair in self.pairs:
                line = json.dumps(data_pair.to_json(), ensure_ascii=False)
                file.write(line + "\n")

    @classmethod
    def load(cls, path: Path):
        data_pairs = []
        with path.open() as file:
            for line in file:
                line_data = json.loads(line)
                data_pair = VADDataPair.from_json(line_data)
                data_pairs.append(data_pair)
        return cls(pairs=data_pairs)
