# -*- coding: utf-8 -*-
# @time      : 2018/12/5 11:08
# @author    : rpyxqi@gmail.com
# @file      : lstm_trading_signal.py


from rqalpha.api import *
import numpy as np
import operator
from collections import defaultdict
from collections import OrderedDict
# import tushare as ts
from quant_models.model_processing.ml_reg_models import Ml_Reg_Model
from rqalpha.core.strategy_context import StrategyContext
from rqalpha.model.portfolio import Portfolio
import pandas as pd
from quant_models.data_processing.data_fetcher import DataFetcher
from quant_models.utils.logger import Logger
from quant_models.data_processing.features_calculation import get_time_series_features
from quant_models.data_processing.features_calculation import get_mkt_features
from quant_models.data_processing.features_calculation import get_source_feature_mappings
from quant_models.model_processing.lstm_models import LSTM_Reg_Model
from quant_models.utils.date_utils import get_month_start_end_date
from quant_models.utils.helper import get_config
from quant_models.utils.logger import Logger
from quant_models.utils.date_utils import get_all_trading_dates
from quant_models.utils.io_utils import load_json_file

data_fetcher = DataFetcher()
config = get_config()

start_date = '20180209'
end_date = '20181012'
pos_num = 10
feature_model_name = 'random_forest'
logger = Logger(log_level='DEBUG', handler='ch').get_log()


def get_sec_codes(trade_date=''):
    # return ['603612.XSHG', '603773.XSHG', '300454.XSHE']
    # mon_start, mon_end = get_month_start_end_date(trade_date)
    ret = load_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\selected_stocks.json')
    return ret.get(trade_date[:-2]) or []
    # return ['000001.XSHE', '000002.XSHE', '000004.XSHE', '000005.XSHE', '000006.XSHE', '000007.XSHE', '000008.XSHE',
    #         '000009.XSHE', '000011.XSHE', '000012.XSHE', '000014.XSHE', '000016.XSHE', '000017.XSHE', '000018.XSHE']


def get_all_features(start_date='', end_date='', step=50, **kwargs):
    features_conf = get_source_feature_mappings(feature_model_name)
    init_cash = kwargs.get('init_cash') or 100000
    trade_date = kwargs.get('trade_date')
    security_ids = get_sec_codes(trade_date)
    cost = kwargs.get('cost') or 0.002
    gamma = kwargs.get('gamma') or 0.0001

    model_config = config['dl_lstm_reg_model']
    features = get_time_series_features(security_ids=security_ids, features=features_conf, start_date=start_date,
                                        end_date=end_date, step=step)
    rows, desc = get_mkt_features(security_ids=security_ids, start_date=start_date, end_date=end_date,
                                  fields=['TURNOVER_VOL', 'CLOSE_PRICE', 'PRE_CLOSE_PRICE'])
    ticker_idx, exchange_idx, pre_close_idx, close_idx, vol_idx, trade_date_idx = desc.index(
        'TICKER_SYMBOL'), desc.index(
        'EXCHANGE_CD'), desc.index('PRE_CLOSE_PRICE'), desc.index('CLOSE_PRICE'), desc.index(
        'TURNOVER_VOL'), desc.index('TRADE_DATE')
    labels = defaultdict(dict)
    params = defaultdict(dict)
    for item in rows:
        security_id = '{0}.{1}'.format(item[ticker_idx], item[exchange_idx])
        return_val = (item[close_idx] - item[pre_close_idx]) / item[pre_close_idx]
        trade_date = item[trade_date_idx].strftime('%Y%m%d')
        labels[security_id].update({trade_date: [return_val]})
        try:
            tmp_param = [init_cash / (item[vol_idx] * item[close_idx])]
            params[security_id].update({trade_date: [init_cash / (item[vol_idx] * item[close_idx])]})
        except Exception as ex:
            logger.error('fail to get mi param for row:{0}, vol idx:{1}, close_idx:{2}, with error:{3}'.format(item, vol_idx,
                                                                                                        close_idx, ex))
            # FIXME hardcode the backup val
            params[security_id].update({trade_date: [0.0001]})
    all_feateures = defaultdict(dict)
    for sec_id, date_val in features.items():
        for date, val in date_val.items():
            curr_feature = [list(item.values()) for item in val]
            try:
                curr_label, curr_param = labels[sec_id][str(date)], params[sec_id][str(date)]
                all_feateures[str(date)][sec_id] = [curr_feature, curr_label, curr_param]
            except Exception:
                logger.info('error to get label and param for sec_code:{0} and date:{1}'.format(sec_id, date))
    return all_feateures


def init(context):
    m = LSTM_Reg_Model('lstm')
    model_config = config['dl_lstm_reg_model']
    m.build_model(step=int(model_config['step']), input_size=int(model_config['input_size']),
                  starter_learning_rate=float(model_config['learning_rate']),
                  hidden_size=int(model_config['hidden_size']),
                  nclasses=int(model_config['nclasses']), decay_step=int(model_config['decay_step']),
                  decay_rate=float(model_config['decay_rate']), cost=0.0001)
    context.model = m
    context.features = get_all_features(start_date, end_date, step=int(model_config['step']),
                                        trade_date=context.now.strftime('%Y%m%d'))


def before_trading(context):
    pass


def handle_bar(context, bar_dict):
    trade_date = context.now.strftime('%Y%m%d')
    # percents_map = {}
    pos_lst = []
    target_features = context.features.get(trade_date)
    if target_features:
        for sec_code, f in target_features.items():
            pos, ret = context.model.predict([f[0]], [f[1]], [f[2]], 0.0001)
            # percents_map.update({sec_code: pos[0]})
            pos_lst.append([sec_code, pos[0][0]])
    # percent_lst = list(percents_map.values())
    pos_lst.sort(key=lambda x: x[1])
    pos_sum = sum([item[1] for item in pos_lst[:pos_num] if not item[1] != item[1]])
    for sec_code, pos in pos_lst[:pos_num]:
        try:
            target_percent = pos / pos_sum
        except Exception as ex:
            logger.error(
                'error for target percent with pos:{0} and pos_sum: {1} with error:{2}'.format(pos, pos_sum, ex))
        try:
            order_target_percent(sec_code, target_percent)
            logger.info('order sec_code:{0} with percent:{1} for date:{2}'.format(sec_code, target_percent, trade_date))
        except Exception as ex:
            logger.error('error for order sec_code:{0}, percent:{1} for trade_date:{2} with error:{3}'.format(sec_code,
                                                                                                              target_percent,
                                                                                                              trade_date,
                                                                                                              ex))


def after_trading(context):
    pass


__config__ = {
    "base": {
        "start_date": "2018-01-09",
        "end_date": "2018-10-12",
        "frequency": "1d",
        "matching_type": "current_bar",
        "data_bundle_path": "rqdata/bundle",
        "benchmark": "000300.XSHG",
        "accounts": {
            "stock": 10000000
        }
    },
    "extra": {
        "log_level": "error",
        "show": True,
    },
    "mod": {
        "sys_progress": {
            "enabled": True,
            "show": True,
        },
    },
}
