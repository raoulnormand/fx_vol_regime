"""
Performs financial computations on the data
"""

# Imports

from fxvol.data_utils import load_csv, save_csv
from fxvol.fin_comp import comp_hist_vol, comp_log_returns

# Interpolate between missing values

df = load_csv("raw", "fx_spots", index_col="Date")
df.interpolate(inplace=True)

# Get log returns and vol

log_returns = comp_log_returns(df)
vol = comp_hist_vol(df)

# Save data

save_csv(df, "processed", "fx_spots")
save_csv(log_returns, "processed", "log_returns")
save_csv(vol, "processed", "historic_vol")
