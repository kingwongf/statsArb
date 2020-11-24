import yaml
from src import backtest
import numpy as np
import pandas as pd
from itertools import combinations
import ray
with open("configs/optimise_trading_rules.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

etf_name = 'xlf'


defactoring='pca'


performance_only = True


weighting_scheme = cfg["weighting_scheme"]
transaction_cost = cfg['transaction_cost']
sl = cfg['sl']
long_only = cfg['long_only']


## xlc ('09-19-2018', '12-31-2019'),
## xlre ('01-06-2016', '12-31-2019'),
## rest ('2007-01-03', '2015-01-02') ##

st_dt, ed_dt = cfg['bt_dt'].get(etf_name, ('2007-01-03', '2015-01-02'))

prices_file_path = cfg["prices_file_path"][etf_name]


m = backtest.bt(prices_file_path=prices_file_path,
       etf_name=etf_name,
       st_dt=st_dt,
       ed_dt=ed_dt, performance_only=performance_only)

# sharpe, maxdd, endpnl = m.run(weighting_scheme=weighting_scheme, sl=-0.10, long_only=False, transaction_cost=transaction_cost, s_threshold=None)

s_thresholds_variations = [{'s_bo': a, 's_bc': b, 's_so': c, 's_sc':d} for a,b,c,d in list(combinations(np.arange(-2.0, 2.0, 0.25),4)) if a<b<c<d]

ray.init()

@ray.remote
def vec_run(s_thresholds):
    return m.run(weighting_scheme=weighting_scheme, sl=sl, long_only=long_only, transaction_cost=transaction_cost, s_thresholds=s_thresholds)

results = [vec_run.remote(s_thresholds=s) for s in s_thresholds_variations]
ray_r = ray.get(results)

r_df = pd.DataFrame(ray_r)
s_thresholds_variations = pd.DataFrame.from_dict(s_thresholds_variations)
r_df = pd.concat([s_thresholds_variations, r_df], axis=1)

r_df.to_csv(f"results/opt_trading_rules/{defactoring}_defactoring/{etf_name}_opt_trading_rules.csv")