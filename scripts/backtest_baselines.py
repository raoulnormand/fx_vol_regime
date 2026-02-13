"""
Compute scores for baseline forecasts.
"""

# Imports


from fxvol.backtest import backtest_results
from fxvol.baseline_models import (
    ewma_forecast,
    garch11_forecast,
    har_forecast,
    naive_forecast,
    rolling_mean_forecast,
)
from fxvol.data_utils import load_csv

# Data

log_ret = load_csv("processed", "log_returns").dropna()
eur_ret = log_ret["EUR"]

# Models

models = [
    (naive_forecast, "naive", {}),
    (rolling_mean_forecast, "rolling5", {"window": 5}),
    (rolling_mean_forecast, "rolling20", {"window": 20}),
    (rolling_mean_forecast, "rolling50", {"window": 50}),
    (rolling_mean_forecast, "rolling100", {"window": 100}),
    (ewma_forecast, "ewma090", {"alpha": 0.9}),
    (ewma_forecast, "ewma030", {"alpha": 0.3}),
    (har_forecast, "har1-5-22", {"lags": [1, 5, 22]}),
    (har_forecast, "har1-5-22-66", {"lags": [1, 5, 22, 66]}),
    (garch11_forecast, "garch11", {}),
]

# Run backest

HORIZON = 5

backtest_results(log_ret=eur_ret, models=models, horizon=HORIZON, file_name="baselines")
