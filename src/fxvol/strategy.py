"""
Vol targeting strategy.
"""

# Imports

import numpy as np
import pandas as pd

from fxvol.backtest import run_backtest
from fxvol.data_utils import save_csv

# Get predictions for a given asset and model


def get_pred(
    data: list,
    model: tuple,
    horizon: int,
    start_date: float | str = 0.5,
) -> pd.DataFrame:
    """
    Get vol predictions for all assets.
    This is just a helper function that runs the backtest.
    'data' is a list of (currency_name, X, y).
    """

    # Unpack model
    forecast_fn, _, kwargs = model

    # Get index of start date
    (_, X, y) = data[0]

    if isinstance(start_date, float):
        end_ix = int(start_date * len(y))
    else:
        end_ix = y.index.get_loc(start_date)

    assert isinstance(end_ix, int)

    # Get prediction on rolling window

    vol_pred = {}

    for currency_name, X, y in data:

        preds = run_backtest(
            X=X,
            y=y,
            horizon=horizon,
            forecast_fn=forecast_fn,
            start_date=start_date,
            stride=horizon,
            **kwargs,
        )

        vol_pred[currency_name] = preds["y_pred"]

    df = pd.DataFrame(data=vol_pred, index=preds.index)
    return df


# Get returns from vol targeting stategy


def run_strategy(
    data: list,
    model: tuple,
    horizon: int,
    target_vol: float,
    nb_trading_days: int = 252,
    start_date: float | str = 0.5,
    file_name: str | None = None,
) -> pd.DataFrame:
    """
    Run the volatility targeting strategy for a given model and returns log returns. Saves file if desired.
    """

    # Make target_vol daily
    daily_target_vol = target_vol / np.sqrt(nb_trading_days)

    # Get prediction
    preds = get_pred(data, model, horizon, start_date)

    # Get log returns over each horizon
    log_rets = pd.DataFrame({curr: X["lr"] for curr, X, _ in data})
    log_rets = log_rets.loc[preds.index[0] :]  # remove burning period
    period_log_rets = log_rets.rolling(horizon).sum()  # returns over horizon

    # Align
    preds = preds.shift(1).iloc[1:]
    period_log_rets = period_log_rets.loc[preds.index]  # remove intermediate days

    # Use exact formula to get log returns over each period, then
    # cumsum to get the total returns
    weights = daily_target_vol / (preds * np.sqrt(horizon))
    portfolio_ret = np.log(
        (weights * np.exp(period_log_rets)).sum(axis=1) + 1 - weights.sum(axis=1)
    )
    # May be better to add a column for riskless asset
    portfolio_ret = portfolio_ret.cumsum()

    # Save results if desired
    if file_name is not None:
        save_csv(portfolio_ret.astype(float), "results", file_name)

    return portfolio_ret
