from pathlib import Path
import click
import pandas as pd


@click.command()
@click.option(
    "-s",
    "--source",
    default="data/sampleSubmission.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def train(source: Path) -> None:
    dataset = pd.read_csv(source)
    click.echo(f"Dataset shape: {dataset.shape}.")