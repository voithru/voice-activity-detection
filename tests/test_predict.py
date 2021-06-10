from pathlib import Path
from tempfile import NamedTemporaryFile

from typer.testing import CliRunner

from main import app
from vad.data_models.voice_activity import VoiceActivity

runner = CliRunner()


def test_predict():
    audio_path = "tests/data/WhenTheWeatherIsFine/When_the_Weather_Is_Fine_12_4.wav"
    checkpoint_path = "tests/checkpoints/vad/sample.checkpoint"

    with NamedTemporaryFile(suffix=".json") as temp_file:
        result = runner.invoke(
            app,
            [
                "predict",
                audio_path,
                checkpoint_path,
                "--output-path",
                temp_file.name,
            ],
        )
        assert result.exit_code == 0

        voice_activity = VoiceActivity.load(Path(temp_file.name))
        assert len(voice_activity.activities) > 0
