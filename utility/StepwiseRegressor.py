import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class StepwiseRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, metric=r2_score):
        self.metric = metric
        self.best_indexes = None
        self.expected_columns_count = None

    def fit(self, X, y, split_data=True):
        available_columns = list(range(X.shape[1]))
        self.expected_columns_count = X.shape[1]
        if split_data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        locked_columns = []
        previous_step_accuracy = -1
        for i in range(X.shape[1]):
            best_accuracy = -1
            best_index = []
            for index in available_columns:
                accuracy = (LinearRegression().fit(X_train[:, [index] + locked_columns], y_train)
                                              .score(X_test[:, [index] + locked_columns], y_test))
                if accuracy > best_accuracy + 0.01:
                    best_accuracy = accuracy
                    best_index = index

            if best_accuracy > previous_step_accuracy:
                previous_step_accuracy = best_accuracy
                locked_columns.append(best_index)
                available_columns.remove(best_index)
            else:
                break
        if len(locked_columns) == 0:
            Warning("Failed to fit stepwise regression. It will use random column instead!")
            locked_columns = [np.random.randint(0, self.expected_columns_count)]
        self.best_indexes = locked_columns
        return self

    def transform(self, X: np.ndarray):
        if self.best_indexes is None:
            raise Exception("Model must be fitted before transformation!")
        if X.shape[1] != self.expected_columns_count:
            raise Exception(f"Columns count are different with fitted data columns!"
                            f"(Input have {X.shape[1]} columns and fitted have {self.expected_columns_count} columns)")
        result = X[:, self.best_indexes]
        return result
