import click
import pandas as pd
import mlflow

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

    def add_num(path=source, num=None):
        if num is None:
            return

        dot_ind = str(path).index('.')
        return str(path)[:dot_ind] + num + str(path)[dot_ind:]

    with mlflow.start_run():
        model1 = create_pipeline(use_scaler=True, model_type='knn', hyperparams={'n_neighbors': 10})
        mlflow.log_param('n_neighbors', 10)

        dump(model1, add_num(target, '1'))
        click.echo(f"Model is saved to {target}.")

        model1_scores = cross_validate(X, y, model1, n_splits=2)

        for metric_name, metric_value in model1_scores.items():
            mlflow.log_metric(metric_name, metric_value)

        click.echo(model1_scores)