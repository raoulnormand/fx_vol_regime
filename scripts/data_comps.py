"""
Performs financial computations on the data
"""

# Imports

from fxvol.data_utils import load_csv, save_csv
from fxvol.fin_comp import comp_log_returns

# Interpolate between missing values

df = load_csv("raw", "fx_spots", index_col="Date")
df.interpolate(inplace=True)

# Get log returns

log_returns = comp_log_returns(df)
log_returns.dropna(inplace=True)

# Save data

save_csv(df, "processed", "fx_spots")
save_csv(log_returns, "processed", "log_returns")
