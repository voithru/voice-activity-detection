from pathlib import Path
from shutil import rmtree

from typer.testing import CliRunner

from main import app

runner = CliRunner()


def test_predict():
    config_path = "tests/configs/vad/train_config.yaml"
    result = runner.invoke(app, ["train", config_path])

    rmtree(Path("results/tests"))
    assert result.exit_code == 0
