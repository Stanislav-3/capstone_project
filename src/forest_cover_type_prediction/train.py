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
@click.option(
    "-m",
    "--model_name",
    default="knn"
)
@click.option(
    "-sc",
    "--use_scaler",
    default=True
)
@click.option(
    "-nn",
    "--n_neighbors",
    default=10
)
@click.option(
    "-p",
    "--penalty",
    default='l2'
)
@click.option(
    "-mi",
    "--max_iter",
    default=1e5
)
@click.option(
    "--tol",
    default=1e-4
)
@click.option(
    "--c",
    default=1.0
)
@click.option(
    "-rs",
    "--random_state",
    default=42
)
def train(
        source: Path,
        target: Path,
        model_name: str,
        use_scaler: bool,
        n_neighbors: int,
        penalty: str,
        max_iter: float,
        tol: float,
        c: float,
        random_state: int
) -> None:
    X, y = get_dataset(source)

    def add_num(path=source, num=None):
        if num is None:
            return

        dot_ind = str(path).index('.')
        return str(path)[:dot_ind] + num + str(path)[dot_ind:]

    with mlflow.start_run():
        hyperparams = {}

        if model_name == 'knn':
            hyperparams['n_neighbors'] = n_neighbors
        else:
            hyperparams['penalty'] = penalty
            hyperparams['max_iter'] = max_iter
            hyperparams['tol'] = tol
            hyperparams['C'] = c
            hyperparams['random_state'] = random_state

        mlflow.log_param('use_scaler', use_scaler)
        for param_name, param_value in hyperparams.items():
            mlflow.log_param(param_name, param_value)

        model = create_pipeline(use_scaler=use_scaler, model_type=model_name, hyperparams=hyperparams)

        dump(model, target)
        click.echo(f"Model is saved to {target}.")

        model_scores = cross_validate(X, y, model, n_splits=2)

        for metric_name, metric_value in model_scores.items():
            mlflow.log_metric(metric_name, metric_value)

        click.echo(model_scores)