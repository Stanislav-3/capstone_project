from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score


def cross_validate(X, y, model, n_splits=2):
    scores = {
        'accuracy_score': 0,
        'precision_score': 0,
        'recall_score': 0
    }

    kf = KFold(n_splits=n_splits)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        scores['accuracy_score'] += accuracy_score(prediction, y_test)
        scores['precision_score'] += precision_score(prediction, y_test, average='micro')
        scores['recall_score'] += recall_score(prediction, y_test, average='micro')

    return {key: value / n_splits for key, value in scores.items()}