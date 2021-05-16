import numpy as np

from utility.StepwiseRegressor import StepwiseRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class LinearGMDH(BaseEstimator, RegressorMixin):
    @staticmethod
    def merge_arrays(a, b):
        return np.array(list(zip(a, b)))

    def __init__(self):
        self.partial_descriptions = None
        self.used_indexes = None

    def fit(self, X: np.ndarray, y: np.ndarray, split_data=True, test_size=0.2):
        if split_data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        partial_descriptions = []
        used_indexes = []
        y_i_train = np.ones(X_train.shape[0])
        y_i_test = np.ones(X_test.shape[0])
        global_best_metric = None
        for i in range(X.shape[1]):
            best_metric = None
            best_index = -1
            for i in set(range(X.shape[1])).difference(set(used_indexes)):
                if i == 1:
                    print(i)
                x_y_train = self.merge_arrays(X[:, i], y_i_train)
                x_y_test = self.merge_arrays(X[:, i], y_i_test)
                partial_description = LinearRegression(fit_intercept=True).fit(x_y_train, y_train)
                metric = partial_description.score(x_y_test, y_test)
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_index = i
            if global_best_metric is None or best_metric > global_best_metric:
                partial_descriptions.append(partial_description)
                used_indexes.append(best_index)
                y_i_train = partial_description.predict(np.array(list(zip(X_train[:, best_index], y_i_train))))
                y_i_test = partial_description.predict(np.array(list(zip(X_test[:, best_index], y_i_test))))
            else:
                break
        self.partial_descriptions = partial_descriptions
        self.used_indexes = used_indexes
        return self

    def transform(self, X):
        Y = self.partial_descriptions[0].predict(self.merge_arrays(X[:, self.used_indexes[0]], np.ones(X.shape[0])))
        for index, partial_description in zip(self.used_indexes[1:], self.partial_descriptions[1:]):
            Y = partial_description.predict(self.merge_arrays(X[:, index], Y))
        return Y

    def __repr__(self):
        result = [
            f"{self.__class__}()"
        ]
        if len(self.used_indexes) > 0:
            result.append("Partial descriptions:")
            a = round(self.partial_descriptions[0].intercept_, 2)
            b = round(self.partial_descriptions[0].coef_[0], 2)
            result.append(f"y_0 = {a} {'+' + str(b) if b > 0 else b} * x_{self.used_indexes[0]}")
            for i, index, partial_description in zip(range(len(self.used_indexes)), self.used_indexes[1:], self.partial_descriptions[1:]):
                a = round(partial_description.intercept_, 2)
                b, c = round(partial_description.coef_[0], 2), round(partial_description.coef_[1], 2)
                result.append(f"y_{i + 1} = {a} {'+' + str(b) if b > 0 else b} * x_{index} {'+' + str(c) if c > 0 else c} * y_{i}")
        return '\n'.join(result) + '\n'
