import click
import pandas as pd
import mlflow

from pathlib import Path
from joblib import dump

from src.forest_cover_type_prediction.data import get_dataset, to_csv
from src.forest_cover_type_prediction.pipeline import create_pipeline, get_model
from src.forest_cover_type_prediction.cross_validation import cross_validate
from src.forest_cover_type_prediction.nested_cross_validation import nested_cross_validate
from sklearn.preprocessing import StandardScaler


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
@click.option(
    "-ncv",
    "--apply_nested_cv",
    default=False
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
        random_state: int,
        apply_nested_cv: bool
) -> None:
    X, y = get_dataset(source)

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

    if apply_nested_cv:
        click.echo("Applying nested CV")
        space={}
        if model_name == 'knn':
            space['n_neighbors'] = [5, 10, 20]
        else:
            space['penalty'] = [penalty, ]
            space['max_iter'] = [max_iter, ]
            space['tol'] = [1e-3, 1e-4]
            space['C'] = [0.2, 1]
            space['random_state'] = [random_state, ]

        model = get_model(model_name)
        if use_scaler:
            X = StandardScaler().fit_transform(X)
        model_scores = nested_cross_validate(X, y, model, space)
        click.echo(model_scores)


def make_submission(train_data_path="data/train.csv", test_data_path="data/test.csv"):
    hyperparams = {
        'penalty': 'l2',
        'max_iter': 1e4,
        'tol': 1e-3,
        'C': 200,
        'random_state': 42
    }

    X_train, y_train = get_dataset(train_data_path)
    X_test, _ = get_dataset(test_data_path)

    model = create_pipeline(use_scaler=True, model_type='linear_regression', hyperparams=hyperparams)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    to_csv(predictions)


