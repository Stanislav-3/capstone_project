import pandas as pd
from pathlib import Path


def get_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv('data/train.csv')

    X = data.drop(['Id'], axis=1).values
    y = data['Cover_Type']

    return X, y

