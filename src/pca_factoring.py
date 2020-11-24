import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA

with open("configs/optimise_trading_rules.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)



def _prep_data():
    full_mkt_data = pd.read_pickle(cfg['full_mkt_data']).ffill()
    mkt_ret_df = full_mkt_data.pct_change()
    return full_mkt_data, mkt_ret_df


def pca_port(args):
    '''
    PCA portfolios returns
    :param args: iterables of (lookback ret_df, current date, current returns, number of PCA components)
    :return:
    '''
    sampled_ret, dt, cur_ret, n_components = args
    rho = sampled_ret.corr()
    sig_bar = sampled_ret.std()[rho.columns]
    v = PCA(n_components=n_components).fit(rho).components_
    v = v / np.sum(np.abs(v))
    f = (v / (sig_bar.values[:, None].T)).dot(cur_ret[rho.columns].values)

    return [dt, *f]
