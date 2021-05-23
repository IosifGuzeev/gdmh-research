import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def get_bool_vectors(size: int):
    result = []
    bool_vector = np.repeat(True, size)
    generate_bool_vectors(bool_vector, size - 1, result)
    return result


def generate_bool_vectors(bool_vector, index, result: list):
    if index < 0:
        result.append(tuple(bool_vector))
        return

    new_vector = bool_vector.copy()
    new_vector[index] = False
    generate_bool_vectors(new_vector, index - 1, result)
    new_vector = bool_vector.copy()
    new_vector[index] = True
    generate_bool_vectors(new_vector, index - 1, result)


class BruteForceOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, metric=r2_score):
        self.metric = metric
        self.best_indexes = None
        self.expected_columns_count = None

    def fit(self, X, y, split_data=True):
        available_columns = np.array((range(X.shape[1])))
        self.expected_columns_count = X.shape[1]
        if split_data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        best_accuracy = -1
        best_index = []
        for bool_vector in get_bool_vectors(X.shape[1]):
            if list(bool_vector).count(True) < 1:
                continue
            model = LinearRegression().fit(X_train[:, bool_vector], y_train)
            accuracy = model.score(X_test[:, bool_vector], y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_index = available_columns[bool_vector]

        if len(best_index) == 0:
            Warning("Failed to fit stepwise regression. It will use random column instead!")
            best_index = [np.random.randint(0, self.expected_columns_count)]
        self.best_indexes = best_index
        return self

    def transform(self, X: np.ndarray):
        if self.best_indexes is None:
            raise Exception("Model must be fitted before transformation!")
        if X.shape[1] != self.expected_columns_count:
            raise Exception(f"Columns count are different with fitted data columns!"
                            f"(Input have {X.shape[1]} columns and fitted have {self.expected_columns_count} columns)")
        result = X[:, self.best_indexes]
        return result
