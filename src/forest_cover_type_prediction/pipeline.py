from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(use_scaler: bool = False, model_type: str = 'knn', hyperparams: dict = {}) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if model_type == 'knn':
        model = KNeighborsClassifier(**hyperparams)
    else:
        model = LogisticRegression(**hyperparams)

    steps.append(("classifier", model))

    return Pipeline(steps=steps)