import timeit

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from models.MIA import Node, Layer
from utility.StepwiseRegressor import StepwiseRegressor


class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_feedback_layer(self, X, y, is_first_layer=True):
        self.is_first_layer = is_first_layer
        if is_first_layer:
            self.fit(X, y)
        else:
            all_nodes = []
            for i in range(1, X.shape[1]):
                all_nodes.append(
                    Node(
                        degree=self.default_params['degree'],
                        include_bias=self.default_params['include_bias']
                    ).fit(X, y, columns=(0, i))
                )
            all_outputs = np.array([node.transform(X) for node in all_nodes]).T
            stepwise_regressor = StepwiseRegressor().fit(all_outputs, y, split_data=self.default_params['split_data'])
            self.nodes = np.array(all_nodes)[stepwise_regressor.best_indexes]
            self.input_dim = X.shape[1]
            self.output_dim = len(self.nodes)
        return self


class KandosNN(BaseEstimator, RegressorMixin):
    @staticmethod
    def __merge_arrays(layer_output, prediction):
        return np.hstack((np.array([prediction]).reshape(-1, 1), layer_output))

    def __init__(self):
        self.layers = []
        self.regressors = []
        self.linear_regressor = LinearRegression()
        self.fit_quality = None
        self.fit_time = 0

    def fit(self, X, y, split_data: bool = True, test_size: float = 0.2):
        start = timeit.timeit()
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if split_data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        eps = 0.01
        new_metric = None
        last_layer_output = X_train
        last_layer_output_test = X_test
        while True:
            if len(self.layers) > 0:
                is_first_layer = False
            else:
                is_first_layer = True
            new_layer = FeedbackLayer().fit_feedback_layer(last_layer_output, y_train, is_first_layer=is_first_layer)
            last_layer_output = new_layer.transform(last_layer_output)
            last_layer_output_test = new_layer.transform(last_layer_output_test)
            new_linear_regressor = LinearRegression().fit(last_layer_output, y_train)
            old_metric = new_metric
            new_metric = new_linear_regressor.score(last_layer_output_test, y_test)
            if old_metric is not None and new_metric < old_metric + eps:
                break
            elif last_layer_output.shape[1] < 3:
                self.linear_regressor = new_linear_regressor
                self.regressors.append(new_linear_regressor)
                self.layers.append(new_layer)
                self.layers[-1].fit_score = new_metric
                break
            else:
                self.linear_regressor = new_linear_regressor
                self.regressors.append(new_linear_regressor)
                self.layers.append(new_layer)
                self.layers[-1].fit_score = new_metric
                last_layer_output = self.__merge_arrays(
                    layer_output=last_layer_output,
                    prediction=new_linear_regressor.predict(last_layer_output)
                )
                last_layer_output_test = self.__merge_arrays(
                    layer_output=last_layer_output_test,
                    prediction=new_linear_regressor.predict(last_layer_output_test)
                )
        self.fit_quality = old_metric if old_metric is not None else old_metric
        end = timeit.timeit()
        self.fit_time = end - start
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        Y = self.layers[0].transform(X)
        for layer, regressor in zip(self.layers[1:], self.regressors[:-1]):
            Y = layer.transform(
                self.__merge_arrays(
                    layer_output=Y,
                    prediction=regressor.predict(Y)
                )
            )
        return self.linear_regressor.predict(Y)

    def __repr__(self, N_CHAR_MAX=700):
        if len(self.layers) == 0:
            return f"{self.__class__}(layers_count=0, NOT FITTED)"
        else:
            return f"{self.__class__}(layers_count={len(self.layers)}, fit_quality={self.fit_quality})"

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
