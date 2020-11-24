import pandas as pd
import numpy as np
import yaml
from functools import reduce
'''
Generate a pickle file of all available historical prices
'''

def read_daily_prices(path):
    df = pd.read_csv(path, index_col=['Date'], parse_dates=True).sort_index()
    df = df[[col for col in df.columns.tolist() if 'Adj. Close' in col]].astype('float64')
    df.columns = [col.replace(' Adj. Close','') for col in df.columns]
    return df

def run(includ_etfs=False):
    with open("configs/optimise_trading_rules.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    prices_file_paths = cfg["prices_file_path"]
    dfs = [read_daily_prices(path) for path in prices_file_paths.values()]

    etf_names = list(map(lambda x:x.upper(),prices_file_paths.keys()))
    return reduce(lambda X, x: pd.merge(X, x, how='left', left_index=True, right_index=True), dfs).sort_index() if \
        includ_etfs else reduce(lambda X, x: pd.merge(X, x, how='left', left_index=True, right_index=True), dfs).drop(etf_names, axis=1).sort_index()
