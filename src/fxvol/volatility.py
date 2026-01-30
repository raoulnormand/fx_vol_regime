"""
Functions dealing with volatility
"""

# Imports

import numpy as np

# Compute historic vol

def hist_vol(df, period = 21, is_log_returns = False):
    """
    Compute the histrorical vol over the given period,
    for each column of the dataframe / series.
    If is_log_returns is False, assumes that the df
    contains prices. If True, assume that the df
    contains log returns.
    """
    # Compute log returns if necessary
    if not is_log_returns:
        log_returns = np.log(df / df.shift(1)).iloc[1:]
    else:
        log_returns = df
    
    # Get vol with a rolling window.
    # Note: current (close) price not included in calculation
    return log_returns.rolling(period, closed = 'left').std()
    
    