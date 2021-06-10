import typer

from vad.evaluate import evaluate_vad_from_scratch
from vad.predict import predict_vad_from_scratch
from vad.train import train_vad_from_scratch

app = typer.Typer()
app.command(name="train")(train_vad_from_scratch)
app.command(name="predict")(predict_vad_from_scratch)
app.command(name="evaluate")(evaluate_vad_from_scratch)

if __name__ == "__main__":
    app()
