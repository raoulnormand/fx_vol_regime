"""
Functions to compute financial quantities
"""

# Imports

import numpy as np
import pandas as pd

# Compute log returns


def comp_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    returns = df / df.shift(1)
    return returns.apply(np.log)


# Compute historic vol


def comp_hist_vol(df: pd.DataFrame, period: int = 21):
    """
    Compute the histrorical vol over the given period,
    for each column of the dataframe / series.
    Current (close) price not included in calculation
    """
    log_returns = comp_log_returns(df)
    return log_returns.rolling(period, closed="left").std()
