import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm
import ray
with open("configs/optimise_trading_rules.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)



def _prep_data():
    full_mkt_data = pd.read_pickle(cfg['full_mkt_data']).ffill()
    mkt_ret_df = full_mkt_data.pct_change()
    return full_mkt_data, mkt_ret_df

ray.init()

@ray.remote
def vec_pca_port(args):
    sampled_ret, dt, ret_dt, n_components = args
    rho = sampled_ret.corr()
    sig_bar = sampled_ret.std()[rho.columns]
    v = PCA(n_components=n_components).fit(rho).components_
    v = v / np.sum(np.abs(v))
    f = (v / (sig_bar.values[:, None].T)).dot(ret_dt[rho.columns].values)

    return [dt, *f]






n_est=60
n_components=15
full_mkt_data, mkt_ret_df = _prep_data()

ret_dfs = [(mkt_ret_df.iloc[i - n_est: i].replace(0,np.nan).dropna(axis=1, how='all').dropna(axis=0, how='all').dropna(axis=1, how='any'),
            mkt_ret_df.iloc[i].name, mkt_ret_df.iloc[i], n_components) for i in range(n_est, mkt_ret_df.shape[0])]

print(ret_dfs)

exit()
results = [vec_pca_port.remote(args) for args in ret_dfs]


ray_r = ray.get(results)



ret_pca_port = pd.DataFrame(ray_r, columns=['Date'] +[f"pca_{n}" for n in range(n_components)]).set_index('Date')

print(ret_pca_port)
# ret_pca_port.to_csv("results/pca_factoring/ret_pca_port.csv")