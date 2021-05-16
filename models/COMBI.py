from utility.StepwiseRegressor import StepwiseRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
import numpy as np


class COMBI(BaseEstimator, RegressorMixin):
    """
    http://www.gmdh.net/GMDH_com.htm
    """
    def __init__(self):
        self.stepwise_regressor = StepwiseRegressor()
        self.linear_regressor = LinearRegression()

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.stepwise_regressor.fit(X, y)
        self.linear_regressor = self.linear_regressor.fit(self.stepwise_regressor.transform(X), y)
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        result = self.stepwise_regressor.transform(X)
        result = self.linear_regressor.predict(result)
        return result

    def predict(self, X):
        return self.transform(X)
