import json
from tempfile import NamedTemporaryFile

from typer.testing import CliRunner

from main import app

runner = CliRunner()


def test_evaluate():
    data_list_path = "tests/data/JamakeSpeechSample/vad-train-sample.jsonl"
    checkpoint_path = "tests/checkpoints/vad/sample.checkpoint"

    with NamedTemporaryFile(suffix=".json") as temp_file:
        result = runner.invoke(
            app,
            [
                "evaluate",
                data_list_path,
                checkpoint_path,
                "--output-path",
                temp_file.name,
            ],
        )
        with open(temp_file.name) as result_file:
            total_result_line = result_file.readline()
            total_result = json.loads(total_result_line)

    assert result.exit_code == 0
    assert total_result["auc"] > 0.1
