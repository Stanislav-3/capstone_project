from pathlib import Path
import click
import pandas as pd

from src.forest_cover_type_prediction.data import get_dataset
from src.forest_cover_type_prediction.pipeline import create_pipeline
from src.forest_cover_type_prediction.cross_validation import cross_validate


@click.command()
@click.option(
    "-s",
    "--source",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def train(source: Path) -> None:
    X, y = get_dataset(source)

    model1 = create_pipeline(use_scaler=True, model_type='knn', hyperparams={'n_neighbors': 10})
    model2 = create_pipeline(use_scaler=True,
                             model_type='logisticregression',
                             hyperparams={
                                 'penalty': 'l2',
                                 'max_iter': 10e4
                             })

    model1_scores = cross_validate(X, y, model1, n_splits=3)
    model2_scores = cross_validate(X, y, model2, n_splits=3)

    click.echo(model1_scores)
    click.echo(model2_scores)
