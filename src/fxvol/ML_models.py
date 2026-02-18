"""
ML-based models.
"""

# Imports

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression

# HAR model: OLS with features = lagged rolling vol
# This should match with the arch package results.


def ols_fc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
) -> float:
    """
    OLS with HAR like features.
    Careful that this is not exactly HAR as we do h steps ahead forecasts
    directly.
    """

    model = LinearRegression()
    model.fit(X_train.iloc[:-1], y_train.iloc[:-1])
    return float(model.predict(X_train.iloc[[-1]]))


def elastic_net_fc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    alpha: float,
) -> float:
    """
    OLS with HAR like features and elastic net regularization.
    """

    model = ElasticNet(alpha=alpha)
    model.fit(X_train.iloc[:-1], y_train.iloc[:-1])
    return float(model.predict(X_train.iloc[[-1]]))


def gb_tree_fc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
) -> float:
    """
    Gradient Boosted tree.
    """

    model = GradientBoostingRegressor()
    model.fit(X_train.iloc[:-1], y_train.iloc[:-1])
    return float(model.predict(X_train.iloc[[-1]]))
