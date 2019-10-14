# -*- coding: utf-8 -*-
# @time      : 2019/9/23 21:12
# @author    : rpyxqi@gmail.com
# @file      : backtest_metrics.py

import pandas as pd
import tushare as ts


def get_annual_profit(p_end, p_start, n):
    # p_end 为最终总资产，p_start 最初总资产，n为回测长度
    # maotai_annual_profit = (1 + (df_maotai.head(1)['close'].values[0] / df_maotai.tail(1)['close'].values[0] - 1)) ** (
    #             250 / df_maotai.shape[0]) - 1
    # geli_annual_profit = (1 + (df_geli.head(1)['close'].values[0] / df_geli.tail(1)['close'].values[0] - 1)) ** (
    #             250 / df_geli.shape[0]) - 1
    # print(u'茅台年化收益: ', maotai_annual_profit, u' 格力电器年化收益: ', geli_annual_profit)
    return (p_end / p_start) ** (250 / n) - 1


def get_beta(prices=[], benchmarks=[]):
    # cov(prices,benchmarks)/sigma^2(benchmarks),prices为每日收益率
    pass


def get_alpha():
    # α = pr−rf−β(Br−rf)
    pass


def get_sharp_ratio(p_r, r_f, sigma_p):
    # p_r策略年化收益率，r_f无风险收益率，sigma_p: 策略收益率波动率
    return (p_r - r_f) / sigma_p


def get_max_down(p_lst=[]):
    pass


def calculate_max_drawdown(code, start='2017-01-01', end='2017-11-21'):
    df = ts.get_hist_data(code, start=start, end=end)
    highest_close = df['close'].max()
    df['dropdown'] = (1 - df['close'] / highest_close)
    max_dropdown = df['dropdown'].max()
    print('max dropdown of %s is %.2f%s' % (code, max_dropdown * 100, '%'))


def calculate_beta(df_stock, df_hs300, code, start='2017-01-01', end='2017-11-20'):
    df = pd.DataFrame({code: df_stock['close'].pct_change(), 'hs300': df_hs300['close'].pct_change()},
                      index=df_stock.index)
    cov = df.corr().iloc[0, 1]
    df_hs300['change'] = df_hs300['close'].pct_change() * 100
    var = df_hs300['change'].var()
    beta = cov / var
    return beta


# 定价曲线
def make_capm(code, start='2017-01-01', end='2017-11-20'):
    df_stock = ts.get_hist_data(code, start=start, end=end)
    df_hs300 = ts.get_hist_data('hs300', start=start, end=end)
    df_stock.sort_index(ascending=True, inplace=True)
    df_hs300.sort_index(ascending=True, inplace=True)
    beta = calculate_beta(df_stock, df_hs300, code, start=start, end=end)
    loss_free_return = 0.04
    df = pd.DataFrame({code: df_stock['close'] / df_stock['close'].values[1] - 1,
                       'hs300': df_hs300['close'] / df_hs300['close'].values[1] - 1,
                       'days': xrange(1, df_stock.shape[0] + 1)}, index=df_stock.index)
    df['beta'] = df['days'] * loss_free_return / 250 + beta * (df['hs300'] - df['days'] * loss_free_return / 250)
    df['alpha'] = df[code] - df['beta']
    df[[code, 'hs300', 'beta', 'alpha']].plot(figsize=(960 / 72, 480 / 72))


df_maotai = ts.get_hist_data(maotai, start=start_date, end=end_date)
get_annual_profit('600519', '000651', '2017-06-01', '2017-11-17')
calculate_max_drawdown('002049')
make_capm('601688')
