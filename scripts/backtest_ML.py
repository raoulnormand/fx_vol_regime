"""
Compute scores for ML forecasts.
"""

# Imports


from fxvol.backtest import backtest_results
from fxvol.data_utils import load_csv
from fxvol.ML_models import har_type_ols_forecast

# Data

log_ret = load_csv("processed", "log_returns").dropna()
eur_ret = log_ret["EUR"]

# Models

models = [
    (har_type_ols_forecast, "ols-1-5-22", {"lags": [1, 5, 22]}),
    (
        har_type_ols_forecast,
        "ols-1-5-22-66",
        {"lags": [1, 5, 22, 66]},
    ),
]

# Run backest

HORIZON = 5

backtest_results(
    log_ret=eur_ret, models=models, horizon=HORIZON, file_name="ML_models", sigfig=7
)
