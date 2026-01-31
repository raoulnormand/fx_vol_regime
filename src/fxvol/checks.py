"""
Function to check df.
May need to be removed or refactored.
"""

import pandas as pd

def check_df(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(columns = ['missing', 'zero'])
    summary['missing'] = df.isna().sum()
    summary['zero'] = (df == 0).sum()
    return summary