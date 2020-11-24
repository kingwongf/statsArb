import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.ou_process import OUProcess
from src.bt_tools import get_position_df, dd, single_stock_stats_arb
from src import bt_tools

class bt(object):
    def __init__(self, prices_file_path, etf_name, st_dt, ed_dt, n_window=60, defactoring='etf', performance_only=False):

        ## ETF and single stocks prices
        df = pd.read_csv(prices_file_path, index_col=['Date'], parse_dates=True).sort_index()
        self.prices_df = df[[col for col in df.columns.tolist() if 'Adj. Close' in col]].astype('float64').ffill()
        self.prices_df.columns = [col.replace(' Adj. Close','') for col in self.prices_df.columns]
        self.ret_df = self.prices_df.pct_change()


        ## PCA portfolios returns
        self.pca_port_ret = pd.read_csv("results/pca_factoring/ret_pca_port.csv", index_col=['Date'], parse_dates=True).sort_index()

        self.st_dt = st_dt
        self.ed_dt = ed_dt
        self.window_st_dt = self.ret_df.iloc[self.ret_df.index.get_loc(st_dt) - n_window].name
        self.n_window = n_window

        self.s_threshold = {'s_bo': -1.25, 's_bc': -0.50, 's_so': 1.25, 's_sc': 0.50 }

        ou_m = OUProcess()
        self.s_scores_df = ou_m.trading_signal_group(df_prices=self.prices_df,
                                                etf_name=etf_name.upper(),
                                                pca_ret=self.pca_port_ret,
                                                n_window=self.n_window,
                                                st_dt=self.st_dt,
                                                ed_dt=self.ed_dt,
                                                defactoring=defactoring)
        self.etf_name = etf_name.upper()

        ## Slice returns df to backtest
        self.bt_ret_df = self.ret_df.loc[self.st_dt: self.ed_dt].iloc[:-1]
        self.performance_only = performance_only



    def run(self, weighting_scheme="equal_weighted", sl=-0.10, long_only=False, transaction_cost=(0.0005, 0.0008), s_thresholds=None):

        ## {'s_bo': -1.25, 's_bc': -0.75, 's_so': 1.25, 's_sc': 0.50 } ##

        s_threshold = self.s_threshold if not s_thresholds else s_thresholds
        long_entry = self.s_scores_df < s_threshold['s_bo']
        long_exit = self.s_scores_df > s_threshold['s_bc']
        short_entry = self.s_scores_df > s_threshold['s_so']
        short_exit = self.s_scores_df < s_threshold['s_sc']


        ## Position dfs
        pos_long_df = get_position_df(long_entry, long_exit, self.ret_df.loc[self.st_dt: self.ed_dt].iloc[:-1], sl_limit=sl)
        pos_short_df = get_position_df(short_entry, short_exit,- self.ret_df.loc[self.st_dt: self.ed_dt].iloc[:-1], sl_limit=sl)



        ## Calculate returns
        long_port_ret =  bt_tools.calc_port_ret(self.bt_ret_df, pos_long_df, epsilon=transaction_cost[0], weighting_scheme=weighting_scheme, direction="long")
        short_port_ret = bt_tools.calc_port_ret(self.bt_ret_df, pos_short_df, epsilon=transaction_cost[0], weighting_scheme=weighting_scheme, direction="short")
        cum_short_port_ret = bt_tools.calc_port_ret(self.bt_ret_df, pos_short_df, epsilon=transaction_cost[0], weighting_scheme=weighting_scheme, direction="cum_short")

        ## Calculate cumulative returns of long and short sides
        port_ret = pd.concat([long_port_ret.rename('long_ret'), short_port_ret.rename('short_ret'), cum_short_port_ret.rename("cum_short_ret"), self.bt_ret_df[self.etf_name]], axis=1).fillna(0)


        port_ret, sharpe, maxdd, endpnl = bt_tools.port_performance(port_ret, long_only=long_only)
        if self.performance_only:
            return [sharpe, maxdd, endpnl]
        else:
            bt_tools.plot_performance(port_ret, self.etf_name, sharpe, maxdd, endpnl)
            return [sharpe, maxdd, endpnl]

        # port_ret[['long_pnl', 'short_pnl']] = (port_ret+ 1).cumprod()
        #
        #
        # ## Assuming Leverage ratio of 2E
        # port_ret['cum_ret'] = (port_ret['long_ret'] - port_ret['short_ret']).fillna(0) if not long_only else port_ret['long_ret']
        # port_ret['cum_pnl'] = (port_ret['cum_ret'] + 1).cumprod()
        # port_ret['max_dd'] = port_ret['cum_pnl'].rolling(126, min_periods=1).apply(dd).dropna()
        #
        # sharpe = np.sqrt(252)*np.mean(port_ret['cum_ret'])/np.std(port_ret['cum_ret'])


        ## Plotting performances
        # fig, axs = plt.subplots(2,2, figsize=(30,20))
        # axs = axs.flatten()
        # port_ret[['long_pnl', 'short_pnl']].plot(ax=axs[0], legend=True)
        #
        # axs[1].plot(port_ret['cum_pnl']) #,legend=True
        #
        # axs[2].hist(port_ret['cum_ret'], label=str(bt_tools.descriptive_stats(port_ret['cum_ret'], round=3)), bins=100) # , legend=True
        #
        # port_ret['max_dd'].plot(ax=axs[3]) # ,legend=True
        #
        # (self.ret_df.loc[self.st_dt: self.ed_dt][etf_name] + 1).cumprod().plot(ax=axs[0], label=etf_name) # ,legend=True
        #
        # axs[0].legend()
        #
        # axs[1].legend([f"Sharpe: {np.round(sharpe,2)} \n "
        #                f"MaxDD: {np.round(min(port_ret['max_dd']),3)} \n "
        #                f"EndPnL: {np.round(port_ret['cum_pnl'].iloc[-1],3)}"], loc='upper left')
        #
        # axs[2].legend()
        #
        #
        # axs[3].legend(["6m rolling MaxDD"])
        # plt.subplots_adjust(left=0.07, bottom=0.10, right=1.0, top=1.0, wspace=0, hspace=0.10)
        # plt.show()
        #
        # plt.close()
        #
        # port_ret.to_pickle("results/port_ret.pkl")

        ## jpm vs xlf
        # fig, axs = plt.subplots(3, 1, sharex='col')
        # jpm_long_ret = (pos_long_df*self.ret_df.loc[self.st_dt: self.ed_dt])['JPM Adj. Close']
        # jpm_short_ret = (pos_short_df*self.ret_df.loc[self.st_dt: self.ed_dt])['JPM Adj. Close']
        # jpm_pnl = (jpm_long_ret.fillna(0) - jpm_short_ret.fillna(0) + 1).cumprod()
        # jpm_s_score = s_scores_df['JPM Adj. Close']
        # jpm_price = self.prices_df.loc[self.st_dt: self.ed_dt]['JPM Adj. Close']
        # jpm_price.plot(ax=axs[0], label='prices')
        # jpm_pnl.plot(ax=axs[1], label='pnl')
        # jpm_s_score.loc[self.st_dt: self.ed_dt].plot(ax=axs[2], label='s_score')
        #
        # for name, level in self.s_threshold.items():
        #     axs[2].axhline(level, lw = 0.5, ls = '--')
        #
        #
        # axs[0].legend(loc='upper right')
        # axs[1].legend(loc='upper right')
        # axs[2].legend(loc='upper right')
        # plt.show()


