from src.ou_process import OUProcess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("data/xlf/xlf_full_hist_2020_11_18.csv", index_col='Date', parse_dates=True).sort_index()
df = df[[col for col in df.columns.tolist() if 'Adj. Close' in col]].astype('float64')
df = df.reindex(sorted(df.columns), axis=1)

pca_port_ret = pd.read_csv("results/pca_factoring/ret_pca_port.csv", index_col=['Date'], parse_dates=True).sort_index()

model = OUProcess()

s_score_df = model.trading_signal_group(df,
                                        etf_name='XLF Adj. Close',
                                        pca_ret=pca_port_ret,
                                        st_dt='2019-11-19',
                                        ed_dt='2020-11-19',
                                        n_window=60,
                                        defactoring='pca')

fig, axes = plt.subplots(11, 6, sharex='col', sharey='row')

for i, ax in enumerate(axes.flat[:s_score_df.shape[1]]):

    s_score_df.iloc[:,i].plot(ax=ax, label=s_score_df.iloc[:,i].name)
    ax.axhline(-1.25, lw=0.5, ls='--')
    ax.axhline(-0.5, lw=0.5, ls='--')
    ax.axhline(0.5, lw=0.5, ls='--')
    ax.axhline(0.75, lw=0.5, ls='--')
    ax.axhline(1.25, lw=0.5, ls='--')
    ax.legend(loc='upper right')

plt.show()