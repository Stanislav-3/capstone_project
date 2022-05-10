import click
import pandas as pd

from pathlib import Path
from joblib import dump

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
@click.option(
    "-t",
    "--target",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def train(source: Path, target: Path) -> None:
    X, y = get_dataset(source)

    model1 = create_pipeline(use_scaler=True, model_type='knn', hyperparams={'n_neighbors': 10})
    model2 = create_pipeline(use_scaler=True,
                             model_type='logisticregression',
                             hyperparams={
                                 'penalty': 'l2',
                                 'max_iter': 10e4
                             })

    def add_num(path, num):
        dot_ind = str(path).index('.')
        return str(path)[:dot_ind] + num + str(path)[dot_ind:]

    dump(model1, add_num(target, '1'))
    dump(model2, add_num(target, '2'))

    model1_scores = cross_validate(X, y, model1, n_splits=2)
    model2_scores = cross_validate(X, y, model2, n_splits=2)

    click.echo(model1_scores)
    click.echo(model2_scores)
