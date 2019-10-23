# -*- coding: utf-8 -*-
# @time      : 2019/1/22 13:22
# @author    : rpyxqi@gmail.com
# @file      : macro_prediction.py

import os
import pandas as pd
import talib as ta
import numpy as np
from quant_models.utils.helper import get_source_root
from quant_models.data_processing.data_fetcher import DataFetcher
from quant_models.utils.date_utils import get_all_month_start_end_dates
from quant_models.model_processing.feature_preprocessing import feature_selection_sort
from quant_models.data_processing.data_fetcher_wd import get_econ_data


def get_market_prediction(start_date='20050103', end_date='20181231', input_period=12, predict_period=6):
    root = get_source_root()
    path = os.path.join(get_source_root(), 'data', 'macro_indicators.xlsx')
    df = pd.read_excel(path)
    econ_ids, cat = list(df['指标ID']), list(df['分项'])
    factor_cats = dict(zip(econ_ids, cat))
    pmi_path = os.path.join(root, 'data', 'features', 'PMI.xls')
    # pmi_rows = pd.read_excel(pmi_path).values
    dates, factors, ids = get_econ_data(start_date=start_date, end_date=end_date)
    df = DataFetcher(source=0)
    rows, desc = df.get_data_fetcher_obj().get_mkt_equd(security_ids=['000300.XSHG'], start_date=start_date,
                                                        end_date=end_date,
                                                        asset_type='idx', fields=['CLOSE_INDEX', 'PRE_CLOSE_INDEX'])

    train_y = []
    ma_lst = []
    # econ_month_lst = [item.strftime('%Y%m') for item in dates]
    # n_pmi = len(pmi_lst)
    # train_x = [pmi_lst[idx: idx + period] for idx in range(n_pmi - period)]
    for item in factors:
        ma_lst.append(ta.SMA(np.array(item, dtype=float), input_period)[input_period - 1:-1])
    ma_lst = np.array(ma_lst).transpose()
    close_idx = desc.index('CLOSE_INDEX')
    pre_close_idx = desc.index('PRE_CLOSE_INDEX')
    trade_date_idx = desc.index('TRADE_DATE')
    monthly_return = {}
    tmp_rows = {}
    mon_dates = get_all_month_start_end_dates(start_date, end_date)

    for item in rows:
        tmp_rows.update({item[trade_date_idx].strftime('%Y%m%d'): item[close_idx]})
    for som, eom in mon_dates:
        k = som[:6]
        v = (tmp_rows.get(eom) - tmp_rows.get(som)) / tmp_rows.get(som)
        monthly_return.update({k: v})
    for m in dates[input_period:]:
        # train_y.append(0 if monthly_return.get(m) < 0 else 1)
        tmp = monthly_return.get(m[:6]) or 0.0
        train_y.append(0.0 if tmp < 0 else 1.0)

    # ma_lst = [[sum(item) / len(item)] for item in train_x]
    factor_names = [factor_cats.get(idx) for idx in ids]
    scores = feature_selection_sort(x=ma_lst, y=train_y, feature_names=factor_names, sort_type='pearson')
    df = pd.DataFrame(scores, columns=['相关性', '指标名'])
    df.to_csv('scores_{0}_months.csv'.format(input_period))
    return scores


if __name__ == '__main__':
    print('4 months:', get_market_prediction(period=12))
    # print('1 year:', get_market_prediction(period=12))
    # print('3 months:', get_market_prediction(period=3))
