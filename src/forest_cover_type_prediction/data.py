import pandas as pd
from pathlib import Path
from typing import Tuple


def get_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)
    y = None

    X = data.drop(['Id'], axis=1)

    if 'Cover_Type' in X.columns:
        y = data['Cover_Type']
        X.drop(['Cover_Type'], axis=1, inplace=True)

    return X.values, y


def to_csv(predictions: pd.Series, source='data/sampleSubmission.csv', target='data/submission.csv') -> None:
    solution = pd.read_csv(source)
    solution['Cover_Type'] = predictions
    solution.to_csv(target, index=False)
