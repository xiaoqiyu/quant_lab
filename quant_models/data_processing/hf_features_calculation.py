# -*- coding: utf-8 -*-
# @time      : 2019/11/12 18:56
# @author    : rpyxqi@gmail.com
# @file      : hf_features_calculation.py

import os
import pandas as pd
from quant_models.utils.helper import get_source_root
from quant_models.data_processing.data_fetcher import DataFetcher

g_db_fetcher = DataFetcher()


# 使用峰度和偏度计算的因子，用1分钟的收益率序列

def get_l2_features(security_id='', trade_date='', start_time='', end_time=''):
    root = get_source_root()
    l2_path = os.path.join(os.path.realpath(root), 'data', 'features', 'level2', trade_date)
    order_path = os.path.join(l2_path, 'l2order', '{0}_{1}.csv'.format(security_id.split('.')[0], trade_date))
    tick_path = os.path.join(l2_path, 'l2tick', '{0}_{1}.csv'.format(security_id.split('.')[0], trade_date))
    trade_path = os.path.join(l2_path, 'l2trade', '{0}_{1}.csv'.format(security_id.split('.')[0], trade_date))
    tick_df = pd.read_csv(tick_path)
    order_df = pd.read_csv(order_path)
    trade_df = pd.read_csv(trade_path)
    print(order_df.shape)
    _start_time_int = int(start_time)
    _end_time_int = int(end_time)

    # _tick_df = tick_df.TrdaeTime>


def get_min_features(security_ids=[], start_date='', end_date='', source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    _min_mkt = _df.get_mkt_mins(startdate=start_date, enddate=end_date, sec_codes=security_ids,
                                table_name='CUST.EQUITY_PRICEMIN')
    print(_min_mkt)

    mkt = Market()
    mkt.sec_code = sec_code
    mkt.exchange = exchange

    # mkt.initialize(start_date='20190601', end_date=end_date, db_obj=db_obj)
    mkt.initialize(start_date=get_prev_trading_date(start_date)[0], end_date=end_date, db_obj=db_obj)


if __name__ == '__main__':
    # get_l2_features(security_id='000651.XSHE', trade_date='20191009', start_time='093000', end_time='100000')
    get_min_features(security_ids=['603612.XSHG'], start_date='20191104', end_date='20191105')
