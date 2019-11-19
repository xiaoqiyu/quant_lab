# -*- coding: utf-8 -*-
# @time      : 2019/11/12 18:56
# @author    : rpyxqi@gmail.com
# @file      : hf_features_calculation.py

import os
import numpy as np
from scipy import stats
from collections import defaultdict
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


def get_tick_l2_features(security_id='', trade_date='', start_time='', end_time=''):
    root = get_source_root()
    l2_path = os.path.join(os.path.realpath(root), 'data', 'features', 'level2', trade_date)
    tick_path = os.path.join(l2_path, 'l2tick', '{0}_{1}.csv'.format(security_id.split('.')[0], trade_date))
    tick_df = pd.read_csv(tick_path)
    _start_time_int = int(start_time) * 1000
    _end_time_int = int(end_time) * 1000
    _tick_df = tick_df[tick_df.Time >= _start_time_int]
    _tick_df = _tick_df[_tick_df.Time < _end_time_int]
    _tick_df.drop_duplicates(keep='last', inplace=True)
    _tick_df[
        'acc_bid_volume5'] = _tick_df.BidVolume1 + _tick_df.BidVolume2 + _tick_df.BidVolume3 + _tick_df.BidVolume4 + _tick_df.BidVolume5
    _tick_df[
        'acc_offer_volume5'] = _tick_df.OfferVolume1 + _tick_df.OfferVolume2 + _tick_df.OfferVolume3 + _tick_df.OfferVolume4 + _tick_df.OfferVolume5
    _tick_df['quote_imbalance5'] = _tick_df['acc_bid_volume5'] - _tick_df['acc_offer_volume5']
    _tick_df['amplitude1'] = (_tick_df['High'] - _tick_df['Low'])/_tick_df['LastPrice']
    print(_tick_df.shape)



def cal_tick_l2_features(df=None, payloads=[], features=[]):
    feature_mappings = {
        'quote_imbalance5': 'a'
    }


def cal_min_ts_features(df=None, payloads=[], features=[]):
    feature_mappings = {
        'ret_var': lambda x: np.var(x['RETURN']),
        'ret_skew': lambda x: stats.skew(x['RETURN']),
        'ret_kurtosis': lambda x: stats.kurtosis(x['RETURN']),
        'total_vol': lambda x: sum(x['VOLUME']),
        'amplitude1': lambda x: (max(x['CLOSEPRICE']) - min(x['CLOSEPRICE'])) / list(x['OPENPRICE'])[0]
    }
    ret = {}
    for f in features:
        ret.update({f: feature_mappings.get(f)(df)})
    return ret


def get_min_features(security_ids=[], start_date='', end_date='', start_time='09:40', end_time='10:30', source=0):
    '''

    :param security_ids:
    :param start_date:
    :param end_date:
    :param start_time: 'mm:ss'
    :param end_time:
    :param source:
    :return:
    '''
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    _min_mkt = _df.get_mkt_mins(startdate=start_date, enddate=end_date, sec_codes=security_ids,
                                table_name='CUST.EQUITY_PRICEMIN')
    df = pd.DataFrame(_min_mkt[0], columns=_min_mkt[1])
    if start_time:
        df = df[df.BARTIME >= start_time]
    if end_time:
        df = df[df.BARTIME < end_time]
    all_dates = set(df['DATADATE'])
    ret = defaultdict(dict)

    close_lst = list(df['CLOSEPRICE'])
    close_lst.insert(0, close_lst[0])
    return_lst = []
    n_row = len(close_lst)
    for idx in range(n_row - 1):
        return_lst.append((close_lst[idx + 1] - close_lst[idx]) / close_lst[idx])
    df['RETURN'] = return_lst
    df['RET2VOL'] = df['RETURN'] / df['VOLUME']
    for d in all_dates:
        _df = df[df.DATADATE == d]
        _date_dict = ret.get(d) or {}
        for sec_id in security_ids:
            ticker, exchangecd = sec_id.split('.')
            _df = _df[_df.TICKER == int(ticker)]
            _ret = cal_min_ts_features(df=_df, features=['ret_var', 'total_vol', 'amplitude1'])
            _date_dict.update(_ret)
        ret[d] = _date_dict
    return ret, df


if __name__ == '__main__':
    ret = get_tick_l2_features(security_id='000651.XSHE', trade_date='20191009', start_time='093000', end_time='100000')
    # ret_features, df = get_min_features(security_ids=['603612.XSHG'], start_date='20191104', end_date='20191105')
    # import pprint
    # pprint.pprint(ret_features)
    # print(df)
