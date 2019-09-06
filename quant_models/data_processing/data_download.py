# -*- coding: utf-8 -*-
# @time      : 2018/12/26 14:24
# @author    : rpyxqi@gmail.com
# @file      : data_download.py

from quant_models.data_processing.features_calculation import get_idx_cons
from quant_models.data_processing.data_fetcher import DataFetcherDB
from quant_models.utils.date_utils import get_all_trading_dates
from quant_models.utils.io_utils import load_json_file
import pprint
import pandas as pd
import uqer
from uqer import DataAPI
from quant_models.utils.date_utils import datetime_delta

c = uqer.Client(token="cae4e8fdd64a6cb9c68e9014ab04fdd823da6c41a77417ce6c8dbdf31db35541")

data_fetcher = DataFetcherDB()


def get_idx_cons_dy(security_id='', index_date=''):
    search_id_mapping = {'000001.XSHG': 1, '000300.XSHG': 1782}
    rows, desc = data_fetcher.get_idx_cons(search_id_mapping.get(security_id))
    ret = []
    code_to_dates = {}
    for item in rows:
        sec_id = '{0}.{1}'.format(item[0], item[1])
        try:
            start_date = item[3].strftime('%Y%m%d')
        except Exception as ex:
            start_date = ''
        try:
            end_date = item[4].strftime('%Y%m%d')
        except Exception as ex:
            end_date = ''
        code_to_dates.update({sec_id: (start_date, end_date)})
    return code_to_dates


def get_cons_by_date(code_to_date_map={}, trade_date=''):
    code_lst = []

    for sec_id, dates in code_to_date_map.items():
        start, end = dates
        if start <= trade_date and (not end or end > trade_date):
            code_lst.append(sec_id)

    return code_lst


def retrieve_factor(start_date='', end_date=''):
    code_dates_mapping = get_idx_cons('000300.XSHG')
    feature_mapping = load_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\feature_mapping.json')
    all_trading_dates = get_all_trading_dates(start_date, end_date)

    #
    # code_lst = get_cons_by_date(code_dates_mapping, d)
    # for f, lst in feature_mapping:
    #     rows, desc = data_fetcher.get_equ_factor(factor_type=f, security_ids=code_lst, start_date=d, end_date=d, fields=lst)
    #     dfs.append(pd.DataFrame(rows, columns=desc))
    testing_mappings = {'return': feature_mapping.get('return')}
    for f, lst in feature_mapping.items():
        dfs = []
        for d in all_trading_dates:
            code_lst = get_cons_by_date(code_dates_mapping, d)
            rows, desc = data_fetcher.get_equ_factor(factor_type=f, security_ids=code_lst, start_date=d,
                                                     end_date=d, fields=lst)
            dfs.append(pd.DataFrame(rows, columns=desc))
        df = dfs[0]
        for data in dfs[1:]:
            df = df.append(data)
        del dfs
        df.to_csv(
            "E:\pycharm\\algo_trading\quant_models\quant_models\data\\features\\{0}_{1}_{2}.csv".format(f, start_date,
                                                                                                        end_date))


def retrieve_announcement(start_date='', end_date=''):
    code_dates_mapping = get_idx_cons('000300.XSHG')
    feature_mapping = load_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\feature_mapping.json')
    all_trading_dates = get_all_trading_dates(start_date, end_date)

    #
    # code_lst = get_cons_by_date(code_dates_mapping, d)
    # for f, lst in feature_mapping:
    #     rows, desc = data_fetcher.get_equ_factor(factor_type=f, security_ids=code_lst, start_date=d, end_date=d, fields=lst)
    #     dfs.append(pd.DataFrame(rows, columns=desc))
    testing_mappings = {'return': feature_mapping.get('return')}
    d = start_date
    from quant_models.utils.date_utils import datetime_delta
    # while d < end_date:
    #     code_lst = get_cons_by_date(code_dates_mapping, d)
    #     rows, desc = data_fetcher.get_equ_factor(factor_type=f, security_ids=code_lst, start_date=d,
    #                                              end_date=d, fields=lst)
    #     dfs.append(pd.DataFrame(rows, columns=desc))
    # df = dfs[0]
    # for data in dfs[1:]:
    #     df = df.append(data)
    # del dfs
    # df.to_csv(
    #     "E:\pycharm\\algo_trading\quant_models\quant_models\data\originals\\{0}_{1}_{2}.csv".format(f, start_date,
    #                                                                                                 end_date))


def get_industry_info():
    rows, desc = data_fetcher.get_industry_info()


if __name__ == '__main__':
    # retrieve_factor(start_date='20160103', end_date='20181225')
    get_industry_info()
