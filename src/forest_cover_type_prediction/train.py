from pathlib import Path
import click
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score

from src.forest_cover_type_prediction.data import get_dataset
from src.forest_cover_type_prediction.pipeline import create_pipeline


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

    model1.fit(X, y)
    model2.fit(X, y)

    prediction1 = model1.predict(X)
    prediction2 = model2.predict(X)

    model1_scores = {
        'accuracy_score': accuracy_score(prediction1, y),
        'precision_score': precision_score(prediction1, y, average='micro'),
        'recall_score': recall_score(prediction1, y, average='micro')
    }

    model2_scores = {
        'accuracy_score': accuracy_score(prediction2, y),
        'precision_score': precision_score(prediction2, y, average='micro'),
        'recall_score': recall_score(prediction2, y, average='micro')
    }

    click.echo(model1_scores)
    click.echo(model2_scores)
