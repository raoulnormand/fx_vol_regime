"""
ML-based models.
"""

# Imports

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# HAR model: OLS with features = lagged rolling vol
# This should match with the arch package results.


def har_type_ols_forecast(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    lags: list[int],
    use_asym: bool = False,
) -> float:
    """
    OLS with HAR like features.
    Careful that this is not exactly HAR as we do h steps ahead forecasts
    directly.
    """
    # Columns for training
    train_cols = ["rv"] + [f"rv_{lag}" for lag in lags if lag != 1]

    if use_asym:
        train_cols.append("asym")

    lr = LinearRegression()
    lr.fit(X_train.iloc[:-1], y_train.iloc[:-1])
    return float(lr.predict(X_train.iloc[[-1]]))
