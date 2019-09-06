# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:23:41 2019

@author: LXF
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_figure1(wind_all_share_index, wind_margin_index):
    """
    绘图融资融券指数及万得全A指数
    """
    f = plt.figure(figsize=(20, 10))
    ax1 = f.add_subplot(211)
    wind_all_share_index.CLOSE.plot(ax=ax1, color='b', use_index=False, linestyle='-.', marker='o', alpha=0.7)
    ax1_1 = ax1.twinx()
    wind_margin_index.CLOSE.plot(ax=ax1_1, color='r', use_index=False, linestyle='-.', marker='v', alpha=0.7)
    ax1.legend(ax1.lines + ax1_1.lines, ['万得全A指数', '融资融券指数'], loc='upper left')
    ax1.set_xticklabels('')
    ax1.set_ylabel('万得全A指数')
    ax1_1.set_ylabel('融资融券指数')
    market_amt = pd.concat([wind_margin_index.AMT / 1e8, (wind_all_share_index.AMT - wind_margin_index.AMT) / 1e8],
                           axis=1)
    market_amt.columns = ['融资融券标的', '非融资融券标的']
    ax2 = f.add_subplot(212)
    ax2.set_ylim([0, 20000])
    ax2.set_ylabel('成交额(亿)')
    ax2_bar1 = ax2.bar(range(market_amt['融资融券标的'].size), (market_amt['融资融券标的']).values, color='r', label='融资融券标的',
                       alpha=0.7, width=0.5)
    ax2_bar2 = ax2.bar(range(market_amt['融资融券标的'].size), (market_amt['非融资融券标的']).values, color='b', label='非融资融券标的',
                       bottom=(market_amt['融资融券标的']).values, alpha=0.7, width=0.5)
    ax2_line1 = ax2.plot(range(market_amt['融资融券标的'].size), (market_amt['融资融券标的']).values, color='g', linestyle='--',
                         marker='o', alpha=0.7)
    ax2_1 = ax2.twinx()
    ax2_1.set_ylim([20, 60])
    ax2_1.set_ylabel('融资融券标的成交额占比')
    ax2_line2 = ax2_1.plot(range(market_amt['融资融券标的'].size),
                           100 * (market_amt['融资融券标的'] / (market_amt['融资融券标的'] + market_amt['非融资融券标的'])).values,
                           color='orange', linestyle='-.', marker='^')
    ax2_1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d%%'))
    ax2.legend([ax2_bar1] + [ax2_bar2] + ax2_line1 + ax2_line2,
               ['融资融券标的成交额', '非融资融券标的成交额', '融资融券标的成交额走势', '融资融券标的成交额占比走势'], loc='upper left')
    plt.xticks(range(market_amt['融资融券标的'].size), [x.strftime('%Y%m%d') for x in market_amt.index])
    for xtick in ax2.get_xticklabels():
        xtick.set_rotation(90)
    f.subplots_adjust(hspace=0)
    return f


def plot_figure2(wind_all_share_index, market_value):
    """
    绘图融资融券规模分析
    """
    #    import matplotlib.dates as mdates
    f = plt.figure(figsize=(20, 10))
    ax1 = f.add_subplot(211)
    wind_all_share_index.CLOSE.plot(ax=ax1, color='b', use_index=False, linestyle='-.', marker='o', ylim=[3000, 4800],
                                    alpha=0.7)
    ax1.set_ylabel('万得全A指数')
    ax1_1 = ax1.twinx()
    (market_value[['FinanceValue', 'SecurityValue']] / 1e8).plot(kind='area', ax=ax1_1, color=['r', 'g'], alpha=0.4,
                                                                 ylim=[7000, 12000], legend=False, use_index=False)
    ax1.legend(ax1.lines + ax1_1.lines, ['万得全A指数', '融资余额', '融券余额'], loc='upper left')
    ax1.set_xticklabels('')
    ax2 = f.add_subplot(212)
    market_value.FinanceValue.pct_change().rolling(window=10).corr(wind_all_share_index.CLOSE.pct_change()).plot(ax=ax2,
                                                                                                                 legend=False,
                                                                                                                 use_index=False,
                                                                                                                 linestyle='-.',
                                                                                                                 marker='o',
                                                                                                                 color='b',
                                                                                                                 alpha=0.7,
                                                                                                                 ylim=[
                                                                                                                     -1,
                                                                                                                     1])
    pd.Series(0.5, index=market_value.index).plot(linestyle='--', color='r', ax=ax2, label=False, use_index=False)
    pd.Series(0, index=market_value.index).plot(linestyle='--', color='orange', ax=ax2, label=False, use_index=False)
    pd.Series(-0.5, index=market_value.index).plot(linestyle='--', color='g', ax=ax2, label=False, use_index=False)
    ax2.set_ylabel('融资余额变化率与万得全A指数变化率相关系数')
    ax2_1 = ax2.twinx()
    (100 * market_value.FinanceBuyValue / wind_all_share_index.AMT).plot(kind='area', color='r', ylim=[6, 15],
                                                                         alpha=0.4, ax=ax2_1, legend=False,
                                                                         use_index=False)
    ax2.legend(ax2.lines + ax2_1.lines,
               ['融资余额变化率与万得全A指数变化率相关系数', '+0.5', '0', '-0.5', '融资买入额占全部A股成交额比列', ], loc='upper left')
    ax2_1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d%%'))
    ax2.set_xlabel('')
    ax2_1.set_ylabel('融资买入额占全部A股成交额比列')
    plt.xticks(range(wind_all_share_index.shape[0]), [x.strftime('%Y%m%d') for x in wind_all_share_index.index])
    for xtick in ax2.get_xticklabels():
        xtick.set_rotation(90)
    f.subplots_adjust(hspace=0)
    return f
