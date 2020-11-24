import pandas as pd
import yaml

with open("configs/optimise_trading_rules.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

prices_file_path = cfg["prices_file_path"]

holdings_path = cfg["holdings_path"]

for etf_name in prices_file_path.keys():
    print(etf_name)
    df = pd.read_csv(prices_file_path[etf_name], index_col=['Date'], parse_dates=True).sort_index()
    df = df[[col for col in df.columns.tolist() if 'Adj. Close' in col]].astype('float64')
    df.columns = [col.replace(' Adj. Close', '') for col in df.columns]

    p_cols = set(df.columns.values.tolist())

    hdf = pd.read_csv(holdings_path[etf_name]) if '.csv' in holdings_path[etf_name] else pd.read_excel(holdings_path[etf_name])

    h_cols = set(hdf['Ticker'].values.tolist())

    print(h_cols.difference(p_cols))
