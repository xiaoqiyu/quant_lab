# -*- coding: utf-8 -*-
# @time      : 2018/12/20 14:51
# @author    : rpyxqi@gmail.com
# @file      : data_visuallize.py


from quant_models.data_processing.data_fetcher import DataFetcher
from quant_models.utils.plot_utils import plot_2D
from quant_models.utils.helper import get_source_root
import os
import pprint
import pandas as pd

data_fetcher = DataFetcher()


def get_dominant_future_contracts():
    rows, desc = data_fetcher.get_data_by_sql(
        "SELECT TICKER_SYMBOL, EXCHANGE_CD,TRADE_DATE,CLOSE_PRICE FROM MKT_FUTD WHERE "
        "TURNOVER_VOL=some(SELECT MAX(turnover_vol) FROM MKT_FUTD WHERE TICKER_SYMBOL LIKE"
        " '%IF%' GROUP BY TRADE_DATE) ORDER BY TRADE_DATE ")
    ticker_idx, close_idx, trade_date_idx = desc.index('TICKER_SYMBOL'), desc.index('CLOSE_PRICE'), desc.index(
        'TRADE_DATE')
    ret = {}
    for item in rows:
        ret[item[trade_date_idx]] = [item[ticker_idx], item[close_idx]]
    return ret


def show_2D_trend():
    new_future_rows, desc = data_fetcher.get_mkt_equd(fields=['CLOSE_PRICE'], security_ids=['IF1812.CCFX'],
                                                      start_date='20180720', end_date='20181219', asset_type='future')
    trade_date_idx, close_idx = desc.index('TRADE_DATE'), desc.index('CLOSE_PRICE')
    y1 = [item[close_idx] for item in new_future_rows]
    x1 = [item[trade_date_idx].strftime('%Y%m%d') for item in new_future_rows]

    new_future_rows, desc = data_fetcher.get_mkt_equd(fields=['CLOSE_INDEX'], security_ids=['000300.XSHG'],
                                                      start_date='20180720', end_date='20181219', asset_type='idx')
    trade_date_idx, close_idx = desc.index('TRADE_DATE'), desc.index('CLOSE_INDEX')
    y2 = [item[close_idx] for item in new_future_rows]
    x2 = [item[trade_date_idx].strftime('%Y%m%d') for item in new_future_rows]
    ret = get_dominant_future_contracts()
    y3 = []
    start_date = '20180720'
    end_date = '20181219'
    for ticker, val in ret.items():
        if start_date <= val[0] < end_date:
            y3.append(val[1])
    total_len = len(y2)
    base = [y2[idx] - y1[idx] for idx in range(total_len)]
    zeros = [0] * total_len
    plot_2D([[x1, y3], [x2, y2], [x1, zeros]], styles=['scatter', 'scatter', 'plot'], legends=['base', '300', ''])


def show_feature_correlate():
    root = get_source_root()
    tops = []
    bottoms = []
    for start_date, end_date in [('20180103', '20181231'), ('20140603', '20160202'), ('20160103', '20171231')]:
        corr_path = os.path.join(os.path.realpath(root), 'conf',
                                 'feature_score_{0}_{1}.csv'.format(start_date, end_date))
        df = pd.read_csv(corr_path)
        df = df.sort_values(by='score', ascending=False)
        df_top = df[df.score > 0.1]
        tops.append(list(df_top['feature']))
        df_bottom = df[df.score < -0.1]
        bottoms.append(list(df_bottom['feature']))
    t = set(tops[0])
    t = t.union(set(tops[1]))
    t = t.union(set(tops[2]))

    b = set(bottoms[0])
    b = b.union(set(bottoms[1]))
    b = b.union(set(bottoms[2]))
    t = t.union(b)
    return list(t)


if __name__ == '__main__':
    ret = show_feature_correlate()
    print(ret)
    # show_2D_trend()
    # ret = get_dominant_future_contracts()
    # x = []
    # y = []
    # for ticker, val in ret.items():
    #     y.append(val[1])
    # x = list(ret.keys())
    # plot_2D([[x,y]], styles=['scatter'])
