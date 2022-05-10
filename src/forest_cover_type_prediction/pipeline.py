from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(use_scaler: bool = False, model_type: str = 'knn', hyperparams: dict = {}) -> Pipeline:
    steps = []

    steps.append(("classifier", model))

    return Pipeline(steps=steps)