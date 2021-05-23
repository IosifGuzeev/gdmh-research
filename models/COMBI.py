from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
import numpy as np
import timeit


from utility.StepwiseRegressor import StepwiseRegressor


class COMBI(BaseEstimator, RegressorMixin):
    """
    http://www.gmdh.net/GMDH_com.htm
    """
    def __init__(self):
        self.optimizer = StepwiseRegressor()
        self.linear_regressor = LinearRegression()
        self.fit_time = 0

    def fit(self, X, y):
        start = timeit.timeit()
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.optimizer.fit(X, y)
        self.linear_regressor = self.linear_regressor.fit(self.optimizer.transform(X), y)
        end = timeit.timeit()
        self.fit_time = end - start
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        result = self.optimizer.transform(X)
        result = self.linear_regressor.predict(result)
        return result

    def predict(self, X):
        return self.transform(X)
