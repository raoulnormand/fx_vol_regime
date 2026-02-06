"""
Define different models.
"""

# Imports

import numpy as np
import pandas as pd

# Naive model


def naive_forecast(log_ret: pd.Series, horizon: int) -> float:
    """
    Naive forecast: predict latest realized vol.
    """
    return log_ret.iloc[-horizon:].std()
