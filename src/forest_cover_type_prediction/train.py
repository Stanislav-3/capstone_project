from pathlib import Path
import click
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


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