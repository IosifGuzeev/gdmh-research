from itertools import combinations
import timeit

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import numpy as np

from utility.StepwiseRegressor import StepwiseRegressor


class Node(BaseEstimator, RegressorMixin):
    def __init__(self, degree=2, include_bias=True):
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        self.stepwise_regressor = StepwiseRegressor()
        self.linear_regressor = LinearRegression()
        self.columns = None

    def fit(self, X, y, split_data: bool = True, test_size: float = 0.2, columns=None):
        self.columns = columns
        if self.columns is None:
            X_ = X.copy()
        else:
            X_ = X[:, columns].copy()
        self.poly_features.fit(X_)
        X_ = self.poly_features.transform(X_)
        # TODO: Add split data for stepwise regressor
        self.stepwise_regressor.fit(X_, y, split_data=split_data)
        X_ = self.stepwise_regressor.transform(X_)
        self.linear_regressor.fit(X_, y)
        return self

    def transform(self, X):
        if self.columns is None:
            result = self.poly_features.transform(X)
        else:
            result = self.poly_features.transform(X[:, self.columns])
        result = self.stepwise_regressor.transform(result)
        result = self.linear_regressor.predict(result)
        return result


class Layer(BaseEstimator, RegressorMixin):
    default_params = {
        'degree': 2,
        'include_bias': True,
        'split_data': True,
        'test_size': 0.2
    }

    def __init__(self, **kwargs):
        self.default_params = {**Layer.default_params, **kwargs}
        self.nodes = []
        self.input_dim = None
        self.output_dim = None
        self.fit_score = None

    def fit(self, X, y):
        all_nodes = []
        for (i, j) in combinations(range(X.shape[1]), r=2):
            all_nodes.append(
                Node(
                    degree=self.default_params['degree'],
                    include_bias=self.default_params['include_bias']
                ).fit(X, y, columns=(i, j))
            )
        all_outputs = np.array([node.transform(X) for node in all_nodes]).T
        stepwise_regressor = StepwiseRegressor().fit(all_outputs, y, split_data=self.default_params['split_data'])
        self.nodes = np.array(all_nodes)[stepwise_regressor.best_indexes]
        self.input_dim = X.shape[1]
        self.output_dim = len(self.nodes)
        return self

    def transform(self, X):
        return np.array([node.transform(X) for node in self.nodes]).T

    def __repr__(self, N_CHAR_MAX=700):
        if self.fit_score is None:
            return f"Layer(input_dim={self.input_dim}, output_dim={self.output_dim})"
        else:
            return f"Layer(input_dim={self.input_dim}, output_dim={self.output_dim}, fit_score={self.fit_score})"


class MIA(BaseEstimator, RegressorMixin):
    """
    http://www.gmdh.net/GMDH_mia.htm
    http://www.gmdh.net/articles/iwim/IWIM_21.pdf
    """

    def __init__(self):
        self.layers = []
        self.linear_regressor = LinearRegression()
        self.fit_quality = None
        self.fit_history = []
        self.fit_time = 0

    def fit(self, X, y, split_data: bool = False, test_size: float = 0.2):
        start = timeit.timeit()
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if split_data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        eps = -0.005
        new_metric = None
        last_layer_output = X_train
        last_layer_output_test = X_test
        while True:
            new_layer = Layer().fit(last_layer_output, y_train)
            last_layer_output = new_layer.transform(last_layer_output)
            last_layer_output_test = new_layer.transform(last_layer_output_test)
            new_linear_regressor = LinearRegression().fit(last_layer_output, y_train)
            old_metric = new_metric
            new_metric = new_linear_regressor.score(last_layer_output_test, y_test)
            if (old_metric is not None and new_metric < old_metric + eps):
                break
            elif last_layer_output.shape[1] < 2:
                self.linear_regressor = new_linear_regressor
                self.layers.append(new_layer)
                self.layers[-1].fit_score = new_metric
                break
            else:
                self.linear_regressor = new_linear_regressor
                self.layers.append(new_layer)
                self.layers[-1].fit_score = new_metric
        self.fit_quality = old_metric if old_metric is not None else old_metric
        end = timeit.timeit()
        self.fit_time = end - start
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        Y = X
        for layer in self.layers:
            Y = layer.transform(Y)
        return self.linear_regressor.predict(Y)

    def __repr__(self, N_CHAR_MAX=700):
        if len(self.layers) == 0:
            return f"MIA(layers_count=0, NOT FITTED)"
        else:
            return f"MIA(layers_count={len(self.layers)}, fit_quality={self.fit_quality})"

    def repr_layers(self):
        result = [
            self.__repr__(),
            "Layers:"
        ]
        for layer in self.layers:
            result.append(repr(layer))
        return '\n'.join(result) + '\n'

    def predict(self, X):
        return self.transform(X)
