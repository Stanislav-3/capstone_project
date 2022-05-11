from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score


def nested_cross_validate(X, y, model, space, inner_n_splits=2, outer_n_splits=2):
    scores = {
        'accuracy_score': 0,
        'precision_score': 0,
        'recall_score': 0
    }

    outer_kf = KFold(n_splits=outer_n_splits)

    for train_index, test_index in outer_kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        search = GridSearchCV(model, space, scoring='accuracy', cv=inner_n_splits, refit=True)

        result = search.fit(X_train, y_train)

        best_model = result.best_estimator_

        prediction = best_model.predict(X_test)

        scores['accuracy_score'] += accuracy_score(prediction, y_test)
        scores['precision_score'] += precision_score(prediction, y_test, average='micro')
        scores['recall_score'] += recall_score(prediction, y_test, average='micro')

    return {key: value / outer_n_splits for key, value in scores.items()}