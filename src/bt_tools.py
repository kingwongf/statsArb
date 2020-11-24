import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def portfolio_equity(e_t, r_it, q_it, q_itdt, s_it, r=0, dt=1/252, d_it=0, episilon=0.0005):
    '''
    calculate portfolio_equity at time t + dt
    :param e_t: E_t, portfolio equity at time t
    :param r_it: R_{it}, stock return, pd.Series with ticker index
    :param q_it: dollar investment in a single stock i at time t, pd.Series with ticker index
    :param q_itdt: dollar investment in a single stock i at time t + dt, pd.Series with ticker index
    :param s_it: price of stock i at time t, pd.Series with ticker index
    :param dt: 1/252
    :param d_it: dividend payable to holders of stock i over the period (t, t+dt), pd.Series with ticker index
    :param episilon: slippage
    :return: portfolio equity
    '''


    return e_t + e_t*r*dt + np.sum(q_it * r_it) - np.sum(q_it) * r * dt + np.sum( q_it * d_it / s_it) - np.sum( abs(q_itdt - q_it))*episilon

def signal_search(df, entry_idx, exit_idx, pos_vec, sl_limit):
    if len(entry_idx)==0:
        return pos_vec

    ## take the first index as first entry
    pos_vec.append((entry_idx[0], 'entry'))

    ## store all exit indices if they are larger than current entry
    exit_idx = [x for x in exit_idx if x > pos_vec[-1][0]]


    if not exit_idx:
        entry_idx =[]

    if exit_idx:
        if sl_limit is not None:
            sl = df.loc[pos_vec[-1][0] +1: exit_idx[0], 'index_return'].cumsum() < sl_limit
            pos_vec.append((sl.idxmax(), 'sl_exit') if sl.any() else (sl.index[-1], 'exit'))
        else:
            ## mark the closest exit
            pos_vec.append((exit_idx[0], 'exit'))

    ## look entry indices that are larger than last exit index
    entry_idx = [x for x in entry_idx if x > pos_vec[-1][0]]

    return signal_search(df,entry_idx, exit_idx, pos_vec, sl_limit)

def pos_vector(df, sl_limit):
    entry_idx = np.where(df['entry']>0)[0].tolist()
    exit_idx = np.where(df['exit']>0)[0].tolist()

    if (not exit_idx) or (not entry_idx):
        return []
    return signal_search(df, entry_idx, exit_idx, [], sl_limit)

def get_position_df(entry_df, exit_df, ret_df, sl_limit=None):
    ''' returns a holding/ position dataframe of each day for each ticker '''
    pos_df = entry_df.copy().reset_index(drop=False)['Date'].to_frame()
    # pos_df = entry_df.copy().reset_index(drop=False)['Date'].to_frame()
    # ret_int_idx_df = ret_df.reset_index(drop=True)


    entry_df, exit_df, ret_int_idx_df = entry_df.astype(int).reset_index(drop=True), exit_df.astype(int).reset_index(drop=True), ret_df.reset_index(drop=True)



    for ticker in entry_df.columns:
        if ticker != 'Date':
            entry_exit_df = pd.concat([entry_df[ticker].rename('entry'), exit_df[ticker].rename('exit'), ret_int_idx_df[ticker].rename('index_return')], axis=1)

            pos_ticker = pd.DataFrame(pos_vector(entry_exit_df, sl_limit=sl_limit), columns=['int_index', 'signal'])

            pos_df[ticker] = pos_ticker.set_index('int_index')['signal'].map({'entry': 1, 'exit': 0})


    return pos_df.ffill().set_index('Date').shift(1).fillna(0)

def dd(ts):
    return np.min(ts / np.maximum.accumulate(ts)) - 1
def _get_eq_w(pos_df):
    return pos_df.div(pos_df.sum(axis=1),axis=0)
def _get_holdings_w(pos_df, full_w= pd.read_csv("data/xlf/xlf_holdings_2020_11_18.csv", usecols=["Ticker", "Weight"], index_col=['Ticker'])['Weight']):
    pos_full_w = pos_df.mul(full_w)
    return pos_full_w.div(pos_full_w.sum(axis=1), axis=0)

def calc_port_ret(ret_df, pos_df, epsilon, weighting_scheme="equal_weighted", direction="long"):
    '''
    Calculating the total return of a weighted portfolio, modified and based on Lectur6 p.3
    :param stocks_ret_df: single stock returns df
    :param pos_df: position df, entry/ exit indicator will be pos_df.diff(1) resulting entry:1 and exit: -1
    :param epsilon: transaction cost in return basis, would be ε * R_{n, n+1}
    :return: pd.Series of total portfolio return

    R_{portfolio} = sum_i^N (R_i * (1 - indicator_{i_entry/exit} * epsilon) * w_i) for period n + 1

    w_i = 1/N' if holding position i else 0 for equal weighted portfolio
    w_i = w_{i_etf_holdings} / sum_i^N' (w_{i_etf_holdings}) for holdings weighted portfolio for i stock in N'

    N: number of stocks
    N': number of stocks which are currently holding,
    n: period n
    w_i: weight of stock i

    However cum short returns should have (1 + indicator_{i_entry/exit} * epsilon) instead as the sign changes when
    we combine the portfolio long_ret - short_ret
    '''

    ## Technically entering the day before
    indicator_entry_exit = pos_df.diff(1).abs()

    ## zero position stays zero while holding stock receive 1/N' weight
    w = _get_eq_w(pos_df) if weighting_scheme=="equal_weighted" else _get_holdings_w(pos_df)


    return (ret_df * (1 - indicator_entry_exit * epsilon) * w).sum(axis=1) if direction != "cum_short" else (ret_df * (1 + indicator_entry_exit * epsilon) * w).sum(axis=1)




def descriptive_stats(ts, round=None):
    d_func = [np.mean, np.std, stats.skew, stats.kurtosis]
    d_name = ['mean', 'std', 'ske', 'kur']
    return {n:f(ts) for n,f in zip(d_name, d_func)} if not round else {n:np.round(f(ts),3) for n,f in zip(d_name, d_func)}


def single_stock_stats_arb(pos_long_df, pos_short_df, stock_price,ret_df, ticker, s_scores_df, s_score_threshold):
    # single stock vs xlf
    assert stock_price.shape[0] == ret_df.shape[0] == s_scores_df.shape[0]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pd.concat([pos_long_df[ticker], pos_short_df[ticker]], axis=1))
    fig, axs = plt.subplots(3, 1, sharex='col', figsize=(15,10))

    stock_long_ret = (pos_long_df*ret_df)[ticker]
    stock_short_ret = (pos_short_df*ret_df)[ticker]

    cum_pnl = (stock_long_ret.fillna(0) - stock_short_ret.fillna(0) + 1).cumprod()

    s_score = s_scores_df[ticker]
    stock_price = stock_price[ticker]
    stock_price.plot(ax=axs[0], label='price')

    cum_pnl.plot(ax=axs[1], label='pnl')
    s_score.plot(ax=axs[2], label='s_score')

    for name, level in s_score_threshold.items():
        axs[2].axhline(level, lw = 0.5, ls = '--')


    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    axs[2].legend(loc='upper right')
    plt.show()
    plt.close()

def port_performance(port_ret_df, long_only=False):
    '''
    Calculate end pnl, sharpe, maxdd from port_ret
    :param port_ret_df: df with 'long_ret' and 'short_ret'
    :return:
    '''
    port_ret_df[['long_pnl', 'short_pnl']] = (port_ret_df[['long_ret','short_ret']] + 1).cumprod()

    ## Assuming Leverage ratio of 2E
    port_ret_df['cum_ret'] = (port_ret_df['long_ret'] - port_ret_df['cum_short_ret']).fillna(0) if not long_only else port_ret_df[
        'long_ret']
    port_ret_df['cum_pnl'] = (port_ret_df['cum_ret'] + 1).cumprod()
    port_ret_df['max_dd'] = port_ret_df['cum_pnl'].rolling(126, min_periods=1).apply(dd).dropna()

    sharpe = np.sqrt(252) * np.mean(port_ret_df['cum_ret']) / np.std(port_ret_df['cum_ret'])
    maxdd = min(port_ret_df['max_dd'])
    endpnl = port_ret_df['cum_pnl'].iloc[-1]
    return port_ret_df, sharpe, maxdd, endpnl

def plot_performance(port_ret_df, etf_name, sharpe, maxdd, endpnl):
    '''
    Plotting portfolio performances
    port_performance, with columns ['long_pnl', 'short_pnl', 'cum_pnl', 'cum_ret', 'max_dd']
    :param port_ret_df:
    :return:
    '''
    ## Plotting performances

    etf_ret = port_ret_df.pop(etf_name)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs = axs.flatten()
    port_ret_df[['long_pnl', 'short_pnl']].plot(ax=axs[0], legend=True)

    axs[1].plot(port_ret_df['cum_pnl'])  # ,legend=True

    axs[2].hist(port_ret_df['cum_ret'], label=str(descriptive_stats(port_ret_df['cum_ret'], round=3)),
                bins=100)  # , legend=True

    port_ret_df['max_dd'].plot(ax=axs[3])  # ,legend=True

    (etf_ret + 1).cumprod().plot(ax=axs[0], label=etf_ret.name)  # ,legend=True

    axs[0].legend()

    axs[1].legend([f"Sharpe: {np.round(sharpe, 2)} \n "
                   f"MaxDD: {np.round(maxdd, 3)} \n "
                   f"EndPnL: {np.round(endpnl, 3)}"], loc='upper left')

    axs[2].legend()

    axs[3].legend(["6m rolling MaxDD"])
    plt.subplots_adjust(left=0.07, bottom=0.10, right=1.0, top=1.0, wspace=0, hspace=0.10)
    plt.show()

    plt.close()