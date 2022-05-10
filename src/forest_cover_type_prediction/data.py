import pandas as pd
from pathlib import Path
from typing import Tuple


def get_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)

    X = data.drop(['Id'], axis=1).values
    y = data['Cover_Type']

    return X, y

