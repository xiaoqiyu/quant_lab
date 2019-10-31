# -*- coding: utf-8 -*-
# @time      : 2018/10/18 13:36
# @author    : rpyxqi@gmail.com
# @file      : feature_calculation.py

import numpy as np
import pandas as pd
import os
import gc
import time
import copy
from quant_models.utils.helper import get_config
from quant_models.utils.helper import get_parent_dir
from quant_models.utils.helper import list_files
from quant_models.utils.helper import adjusted_sma
from quant_models.utils.helper import get_source_root
from collections import defaultdict
from collections import OrderedDict
from quant_models.data_processing.data_fetcher import DataFetcher
from quant_models.utils.io_utils import load_json_file
from quant_models.utils.io_utils import write_json_file
from quant_models.utils.logger import Logger
from quant_models.utils.sql_lite_helper import SQLiteHelper

numerical_default = 0.0
config = get_config()

logger = Logger('log.txt', 'DEBUG', __name__).get_log()

_sgn = lambda val: 1 if val >= 0 else -1
# FIXME TO CHECK THE close of connection
g_db_fetcher = DataFetcher()


def get_idx_cons_dy(security_id='', index_date=None, source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    if source == 1:
        ticker = security_id.split('.')[0]
        return _df.get_idx_cons_dy(ticker=ticker, index_date=index_date)
    search_id_mapping = {'000001.XSHG': 1, '000300.XSHG': 1782, '000016.XSHG': 28, '399006.XSHE': 3326} or security_id
    rows, desc = _df.get_idx_cons_dy(idx_id=search_id_mapping.get(security_id), index_date=index_date)
    return ['{0}.{1}'.format(item[0], item[1]) for item in rows]


def get_idx_adjust_dates(security_id='', start_date='', end_date='', source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    if source == 1:
        # TODO TO BE ADDED
        pass
    idx_id = {'000001.XSHG': 1, '000300.XSHG': 1782, '000016.XSHG': 28, '399006.XSHE': 3326}.get(security_id)
    rows, desc = _df.get_adjust_dates(idx_id, start_date, end_date)
    return [item[0].strftime('%Y%m%d') for item in rows]


def get_idx_weights(security_id='', index_date=None, source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    if source == 1:
        # TODO TO BE ADDED
        pass
    idx_id = {'000001.XSHG': 1, '000300.XSHG': 1782, '000016.XSHG': 28, '399006.XSHE': 3326}.get(security_id)
    rows, desc = _df.get_idx_weights(index_date, idx_id)
    return dict(zip(['{0}.{1}'.format(item[0], item[1]) for item in rows], [item[2] for item in rows]))


def get_idx_cons_jy(idx_security_id='', start_date='', end_date='', source=0):
    idx_security_id = idx_security_id.split('.')[0]
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_idx_cons_jy(idx_security_id=idx_security_id)
    inner_codes = []
    for row in rows:
        if row[1].strftime('%Y%m%d') <= start_date:
            if isinstance(row[2], pd.tslib.NaTType):
                inner_codes.append(row[0])
            elif row[2].strftime('%Y%m%d') >= end_date:
                inner_codes.append(row[0])
            else:
                pass
    rows, cols = _df._get_sec_main(inner_codes=inner_codes)
    return ['{0}.{1}'.format(item[0], 'XSHG' if item[1] == 83 else 'XSHE') for item in rows]


def get_idx_adjust_dates_jy(idx_security_id='', start_date='', end_date='', source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_idx_cons_jy(idx_security_id=idx_security_id.split(',')[0])
    dates = []
    for row in rows:
        if row[1].strftime('%Y%m%d') >= start_date and row[1].strftime('%Y%m%d') <= end_date:
            dates.append(row[1])
    dates.append(start_date)
    dates.append(end_date)
    return sorted(dates)


def get_sw_2nd_indust(security_ids=[], source=0):
    '''
    Retrived from datayes database
    :param security_ids:
    :param source:
    :return:
    '''
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_sw_2nd(security_ids=security_ids)
    return dict(zip(['{0}.{1}'.format(item[0], item[1]) for item in rows], [item[2] for item in rows]))


def _resolve_start_end_dates(start_date=None, end_date=None, datetime_intervals=[]):
    if start_date and end_date:
        return start_date, end_date
    dates = []
    for item in datetime_intervals:
        try:
            sd, ed = item[0].split(' ')[0], item[1].split(' ')[0]
            dates.extend([sd, ed])
        except Exception as ex:
            logger.error('Datetime format not valid for {0} with error'.format(item, ex))
    if dates:
        dates = list(set(dates))
        dates = sorted(dates)
        return dates[0], dates[-1]


def save_features(columns=(), payload=[]):
    '''

    :param payload: {'key1':[val_lst]}
    :return:
    '''
    sql_db = SQLiteHelper()
    sql_str = "INSERT INTO FEATURE_CACHE ({0}) VALUES ".format(','.join(columns))
    is_first = True
    for row in payload:
        if is_first:
            is_first = False
            _str = ','.join(row)
            sql_str = '{0} ({1})'.format(sql_str, _str)
        if not is_first:
            sql_str = "{0},".format(sql_str)
            _str = ','.join(row)
            sql_str = '{0} ({1})'.format(sql_str, _str)
    print(sql_str)
    sql_db.execute_sql(sql_str)


def read_features(feature_name=None):
    sql_db = SQLiteHelper()
    return sql_db.execute_query("SELECT * FROM FEATURE_CACHE")


# TODO resolve the case when the source features are derived from the market data
# FIXME check the logic and add calculate feature logic in this function
def get_cal_features(source_feature_dict={}, cal_features=[]):
    return source_feature_dict
    # if not cal_features:
    #     return source_feature_dict
    #
    # def _cal_testing_feature2(val):
    #     pass
    #
    # feature_cal_mappings = {
    #     "testing_feature1": lambda val: val.get('key1') + val.get('val2'),
    #     "testing_feature2": _cal_testing_feature2,
    # }
    # input_val = copy.deepcopy(source_feature_dict)
    # for item in cal_features:
    #     try:
    #         source_feature_dict.update({item: feature_cal_mappings.get(item)(input_val)})
    #     except Exception as ex:
    #         logger.error(
    #             "Fail to calculate the feature {0} for param:{1} with error:{2}".format(item, source_feature_dict, ex))
    #
    # del input_val
    # return source_feature_dict


def get_equity_daily_features(security_ids=[], features={'ma': ['ACD6', 'ACD20']}, start_date=20181101,
                              end_date=20181102,
                              trade_date=None, source=0):
    logger.info(
        'Start calculate features from {0} to {1} for sec_ids:{2} and features types{3}'.format(start_date, end_date,
                                                                                                len(security_ids),
                                                                                                len(features)))

    ret_features = defaultdict(dict)
    # query on one date
    if trade_date:
        start_date = end_date = trade_date
    if isinstance(start_date, str):
        start_date = int(start_date)
    if isinstance(end_date, str):
        end_date = int(end_date)
    retrieve_feature_names = list()
    for f_type, f_val in features.items():
        retrieve_feature_names.extend(f_val)
    retrieve_feature_names = list(set(retrieve_feature_names))
    root = get_source_root()
    feature_mapping = load_json_file(os.path.join(os.path.realpath(root), 'conf', 'feature_mapping.json'))
    source_features = []
    for item in list(feature_mapping.values()):
        source_features.extend(item)
    cal_features = list(set(retrieve_feature_names) - set(source_features))
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    excluded = ['CREATE_TIME', 'UPDATE_TIME', 'TMSTAMP', 'ID', 'SECURITY_ID_INT', 'SECURITY_ID', 'TRADE_DATE',
                'TICKER_SYMBOL']
    retrieve_feature_names = [item.upper() for item in retrieve_feature_names]
    for f_type, f_fields in features.items():
        rows, desc = _df.get_equ_factor(fields=f_fields, factor_type=f_type, security_ids=security_ids,
                                        start_date=start_date, end_date=end_date)
        # logger.info('Complete querying')
        id_idx = desc.index('SECURITY_ID')
        date_idx = desc.index('TRADE_DATE')
        # logger.info('start processing rows for factor type:{0}'.format(f_type))

        if not f_fields:
            # TODO add other fileds
            retrieve_feature_names.extend(list(set(desc) - set(excluded)))
            retrieve_feature_names = [item.upper() for item in retrieve_feature_names]
        for item in rows:
            sec_id, date = item[id_idx], item[date_idx]
            date_dict = ret_features[date] or {}
            if date_dict and sec_id in date_dict:
                curr_dict = date_dict[sec_id]
            else:
                curr_dict = {}
                date_dict[sec_id] = {}
            idx_lst = []
            for idx, val in enumerate(desc):
                if val.upper() in retrieve_feature_names:
                    idx_lst.append(idx)
            # idx_lst = [idx for idx, val in enumerate(desc) if val.upper() in retrieve_feature_names]
            tmp_lst = [item[idx] for idx in idx_lst]
            keys = f_fields or desc
            tmp_dict = dict(zip([desc[idx] for idx in idx_lst], tmp_lst))
            tmp_dict1 = copy.deepcopy(tmp_dict)
            for k1, v1 in tmp_dict1.items():
                if k1 in excluded:
                    tmp_dict.pop(k1)
            # add the pre-defined calculated featuers
            tmp_dict = get_cal_features(tmp_dict, cal_features)
            if tmp_dict:
                curr_dict.update(tmp_dict)
            if curr_dict:
                ret_features[date][sec_id] = curr_dict
        # logger.info('complete processing rows for factor type:{0}'.format(f_type))
        del rows
        gc.collect()
        time.sleep(3)
    for date, val in ret_features.items():
        for sec_id, _val in val.items():
            _keys = set(_val.keys())
            _add_keys = set(retrieve_feature_names) - _keys
            _remove_keys = _keys - set(retrieve_feature_names)
            for _k in _add_keys:
                _val.update({_k: None})
            for _k in _remove_keys:
                _val.pop(_k)
    # FIXME check whether the length of the features are the same now
    return ret_features


def get_time_series_features(security_ids=[], features={'ma': ['ACD6', 'ACD20']}, start_date='', end_date='', step=10):
    ret_features = get_equity_daily_features(security_ids=security_ids, features=features, start_date=start_date,
                                             end_date=end_date)
    time_series_features = defaultdict(dict)
    for sec_code, tmp_dict in ret_features.items():
        logger.info('processing sec_id:{0} in time series features'.format(sec_code))
        ordered_dict = dict(OrderedDict(sorted(tmp_dict.items())))
        dates = list(ordered_dict.keys())
        feature_lst = list(ordered_dict.values())
        for idx, date in enumerate(dates):
            if idx >= step:
                time_series_features[sec_code][date] = feature_lst[idx - step: idx]
    return time_series_features


def get_equity_returns(security_ids=[], start_date='20181101', end_date='20181103', source=0, trade_date=None):
    if trade_date:
        start_date = end_date = trade_date
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_mkt_equd(security_ids=security_ids, fields=['CLOSE_PRICE', 'PRE_CLOSE_PRICE'],
                                  start_date=start_date, end_date=end_date)
    ticker_idx, exchange_idx, pre_close_idx, close_idx, trade_date_idx = desc.index('TICKER_SYMBOL'), desc.index(
        'EXCHANGE_CD'), desc.index('PRE_CLOSE_PRICE'), desc.index('CLOSE_PRICE'), desc.index('TRADE_DATE')
    ret_label = defaultdict(dict)
    for item in rows:
        security_id = '{0}.{1}'.format(item[ticker_idx], item[exchange_idx])
        return_val = (item[close_idx] - item[pre_close_idx]) / item[pre_close_idx]
        trade_date = item[trade_date_idx].strftime('%Y%m%d')
        ret_label[security_id].update({trade_date: return_val})
    return ret_label


def get_market_value(security_ids=[], start_date='20181101', end_date='20181103', source=0, trade_date=None):
    if trade_date:
        start_date = end_date = trade_date
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_mkt_equd(security_ids=security_ids, fields=['NEG_MARKET_VALUE'],
                                  start_date=start_date, end_date=end_date)
    ticker_idx, exchange_idx, mv_idx, trade_date_idx = desc.index('TICKER_SYMBOL'), desc.index(
        'EXCHANGE_CD'), desc.index('NEG_MARKET_VALUE'), desc.index('TRADE_DATE')
    ret_label = defaultdict(dict)
    for item in rows:
        security_id = '{0}.{1}'.format(item[ticker_idx], item[exchange_idx])
        trade_date = item[trade_date_idx].strftime('%Y%m%d')
        ret_label[security_id].update({trade_date: item[mv_idx]})
    return ret_label


def get_idx_returns(security_ids=[], start_date='20181101', end_date='20181103', source=0, trade_date=None):
    if trade_date:
        start_date = end_date = trade_date
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_mkt_equd(security_ids=security_ids, fields=['CHG_PCT'],
                                  start_date=start_date, end_date=end_date, asset_type='idx')
    ticker_idx, exchange_idx, chg_pct_idx, trade_date_idx = desc.index('TICKER_SYMBOL'), desc.index(
        'EXCHANGE_CD'), desc.index('CHG_PCT'), desc.index('TRADE_DATE')
    ret_label = defaultdict(dict)
    for item in rows:
        security_id = '{0}.{1}'.format(item[ticker_idx], item[exchange_idx])
        trade_date = item[trade_date_idx].strftime('%Y%m%d')
        ret_label[security_id].update({trade_date: float(item[chg_pct_idx])})
    return ret_label


def get_mkt_features(security_ids=[], start_date='20181101', end_date='20181103',
                     fields=['TURNOVER_VOL', 'CLOSE_PRICE'], source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    return _df.get_mkt_equd(security_ids=security_ids, fields=fields,
                            start_date=start_date, end_date=end_date)


def get_announcement_profitability_features(stock_ids=[], start_date='', end_date='', source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_annoucement_profitability(stock_ids=stock_ids, start_date=start_date,
                                                   end_date=end_date)
    stockid_idx, category_idx, profitability_idx, pubdate_idx = desc.index('STOCKID'), desc.index(
        'ZS_AUTO_CATEGORY'), desc.index('PROFITABILITY'), desc.index('PUBLISH_DATE')
    ret_prob = defaultdict(dict)
    for item in rows:
        stock_id = item[stockid_idx]
        pub_date = item[pubdate_idx].strftime('%Y%m%d')
        prob = item[profitability_idx]
        cat = item[category_idx]
        prob_flag = 0
        if prob == u'利空':
            prob_flag = 1
        elif prob == u'利好':
            prob_flag = 2
        else:
            prob_flag = 0
        ret_prob[stock_id].update({pub_date: prob_flag})
    return ret_prob


def get_ma(security_ids=[], start_date='', end_date='', period=10, asset_type='idx', source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    price_field = 'CLOSE_INDEX' if asset_type == 'idx' else 'CLOSE_PRICE'
    rows, desc = _df.get_mkt_equd(security_ids=security_ids, fields=[price_field],
                                  start_date=start_date, end_date=end_date, asset_type='idx')
    # TODO to be completed
    vals = defaultdict(list)
    ticker_idx, exchangecd_idx, close_idx, trade_date_idx = desc.index('TICKER_SYMBOL'), desc.index(
        'EXCHANGE_CD'), desc.index(price_field), desc.index('TRADE_DATE')
    for item in rows:
        sec_id = '{0}.{1}'.format(item[ticker_idx], item[exchangecd_idx])
        vals[sec_id].append([item[trade_date_idx], item[close_idx]])
    ret = defaultdict(dict)
    for sec_id, val in vals.items():
        val.sort(key=lambda x: x[0])
        total_len = len(val)
        if total_len < period:
            continue
        lst = [item[1] for item in val]
        ma_lst = adjusted_sma(lst, period)
        ret[sec_id] = dict(zip([item[0].strftime('%Y%m%d') for item in val], ma_lst))
    return ret


def get_halt_security_ids(security_ids=[], curr_date='', source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_halt_info(security_ids=security_ids)
    ret = []
    for item in rows:
        try:
            start_date = item[2].strftime('%Y%m%d') if item[2] else None
            end_date = item[3].strftime('%Y%m%d') if item[3] else None
            if start_date <= curr_date and not end_date or (start_date <= curr_date and curr_date <= end_date):
                ret.append('{0}.{1}'.format(item[0], item[1]))
        except Exception as ex:
            logger.error('Fail to calculate with error:{0}'.format(ex))
    return list(set(ret))


def get_significant_features(top_ratio=0.5, bottom_ratio=0.2):
    root = get_source_root()
    tops = []
    bottoms = []
    ret = defaultdict(dict)
    score_lst = []
    f_types = set()

    root = get_source_root()
    feature_mapping = load_json_file(os.path.join(os.path.realpath(root), 'conf', 'feature_mapping.json'))
    # for start_date, end_date in [('20180103', '20181230'), ('20140603', '20160103'), ('20160103', '20171230')]:
    for start_date, end_date in [('20150103', '20181231')]:
        corr_path = os.path.join(os.path.realpath(root), 'conf',
                                 'score_{0}_{1}.csv'.format(start_date, end_date))
        df = pd.read_csv(corr_path)
        score_ret = defaultdict(list)
        for idx, k, s, ft in df.values:
            _tmp = score_ret.get(ft) or list()
            _tmp.append([k, s])
            score_ret.update({ft: _tmp})
            f_types.add(ft)
        for ft, val in score_ret.items():
            logger.debug(ft, len(val))
            val.sort(key=lambda x: x[1], reverse=True)
            t_dict = ret.get(ft) or dict()
            top_dict = t_dict.get('top_features') or list()
            bottom_dict = t_dict.get('bottom_features') or list()
            top_idx = int(len(val) * top_ratio) if int(len(val) * top_ratio) > 5 else 5
            bottom_idx = int(len(val) * bottom_ratio) if int(len(val) * bottom_ratio) > 1 else 1
            top_dict.extend(item[0] for item in val[:top_idx])
            bottom_dict.extend(item[0] for item in val[-bottom_idx:])
            t_dict.update({'top_features': top_dict})
            t_dict.update({'bottom_features': bottom_dict})
            ret.update({ft: t_dict})
    for ft, val in ret.items():
        top_lst = list(set(val['top_features']))
        bottom_lst = list(set(val['bottom_features']))
        top_lst.extend(bottom_lst)
        ret.update({ft: top_lst})
    return ret


def feature_refine():
    root = get_source_root()
    feature_mapping = load_json_file(os.path.join(os.path.realpath(root), 'conf', 'feature_mapping.json'))
    score_file = 'testing_train_features_score_20160103_20171230.csv'
    score_file = os.path.join(os.path.realpath(root), 'conf', score_file)
    df = pd.read_csv(score_file)
    values = df.values
    type_lst = []
    for item in values:
        f, s = item[1:]
        for k, v in feature_mapping.items():
            if f in v:
                type_lst.append(k)
    df['feature_type'] = type_lst
    save_file = os.path.join(os.path.realpath(root), 'conf', 'score_20160103_20171230.csv')
    df.to_csv(save_file)


def get_source_feature_mappings(feature_types=None):
    root = get_source_root()
    feature_mapping = load_json_file(os.path.join(os.path.realpath(root), 'conf', 'feature_mapping.json'))
    if not feature_types:
        return feature_mapping
    for k, v in feature_mapping.items():
        if not k in feature_types:
            feature_mapping.pop(k)
    # return feature_mapping
    return {'growth': ['TOTALASSETGROWRATE'], 'vs': ['PE', 'PB', 'PS'], 'volume': ['OBV'],
            'return': ['Variance20', 'Alpha20', 'Beta20']}


def get_security_codes(sec_type=1, source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_security_codes(sec_type)
    ticker_idx = desc.index("SecuCode")
    sec_mkt_idx = desc.index("SecuMarket")
    return ['{0}.{1}'.format(item[ticker_idx], 'XSHG' if item[sec_mkt_idx] == 83 else 'XSHE') for item in rows]


def get_sw_indust(source=0):
    '''
    Retrived from jy database, return the 1st level industry code, could be extended to 2dn and 3rd level code and name
    :param source:
    :return:
    '''
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    rows, desc = _df.get_sw_indust()
    ticker_idx = desc.index("SecuCode")
    mkt_idx = desc.index("SecuMarket")
    first_code_idx = desc.index("FirstIndustryCode")
    ret = {}
    for row in rows:
        ret.update({"{0}.{1}".format(row[ticker_idx], "XSHE" if row[mkt_idx] == 90 else "XSHG"): row[first_code_idx]})
    return ret


def get_indust_mkt(start_date=None, end_date=None, source=0):
    _df = g_db_fetcher.get_data_fetcher_obj(source)
    _rows, cols = _df.get_sw_idx_codes_jy()
    inner_codes = [item[cols.index('IndexCode')] for item in _rows]
    industry_codes = [item[cols.index('IndustryCode')] for item in _rows]
    industry_names = [item[cols.index('ChiName')] for item in _rows]
    _code_mapping = dict(zip(inner_codes, industry_codes))
    _name_mapping = dict(zip(inner_codes, industry_names))
    df_mkt = _df.get_indust_mkt_jy(idx_codes=list(set(inner_codes)), start_date=start_date, end_date=end_date,
                                   return_df=True)
    df_mkt['IndustryCode'] = [_code_mapping.get(k) for k in df_mkt['InnerCode']]
    df_mkt['IndustryName'] = [_name_mapping.get(k) for k in df_mkt['InnerCode']]
    df_mkt = df_mkt[['IndustryCode', 'TradingDay', 'ChangePCT', 'IndustryName']]
    return df_mkt


if __name__ == '__main__':
    import pprint

    # ret = get_feature(security_ids=['000001.XSHE'], features={'volume': [], 'ma': []},
    #                   start_date='20160101', end_date='20190415')
    # cols = []
    # rows = []
    # for sec_code, val1 in ret.items():
    #     for date, val2 in val1.items():
    #         # cols.extend(val2.keys())
    #         cols = list(val2.keys())
    #         line = [date] + list(val2.values())
    #         rows.append(line)
    # import pandas as pd
    # cols.insert(0, 'DATE')
    # df = pd.DataFrame(rows, columns=cols)
    # df.to_csv('sample_data.csv')

    # pprint.pprint(ret)
    # ret = get_equity_returns(security_ids=['000001.XSHE'], start_date='20181101', end_date='20181110')
    # pprint.pprint(ret)
    # ret = get_mkt_features(security_ids=['000001.XSHE'], start_date='20181101', end_date='20181110',
    #                        fields=['TURNOVER_VOL', 'CLOSE_PRICE', 'PRE_CLOSE_PRICE'])
    # pprint.pprint(ret)
    # ret = get_announcement_profitability_features(stock_ids=['300100', '601717'], start_date='20170101',
    #                                               end_date='20181102')
    # pprint.pprint(ret)

    # ret = get_ma(security_ids=['000001.XSHG'], start_date='20180101', end_date='20181110')
    # pprint.pprint(ret)

    # ret = get_halt_security_ids(security_ids=['300659.XSHE'], curr_date='20181219')
    # print(ret)

    # ret = get_idx_cons(security_id='000016.XSHG', index_date='20181210')
    # pprint.pprint(ret)

    # ret = get_source_feature_mappings(train_feature=False)
    # s = 0
    # ret1 = []
    # for k, v in ret.items():
    #     ret1.extend(v)
    # print(len(ret1), len(list(set(ret1))))
    # pprint.pprint(ret)

    # ret = get_idx_cons(security_id='399006.XSHE', index_date='20190103', source=1)
    # print('300083.XSHE' in ret)
    # print('300364.XSHE' in ret)
    # print('300134.XSHE' in ret)
    # print(len(ret))
    # ret = get_significant_features()
    # pprint.pprint(ret)
    # feature_refine()

    # ret = get_source_feature_mappings(train_feature=False)
    # pprint.pprint(ret)

    # ret = get_idx_returns(['000300.XSHG'])
    # print(ret)
    # ret = get_significant_features(top_ratio=0.25, bottom_ratio=0.2)

    # print([(k, len(v)) for k, v in ret.items()])
    # print(sum([len(v) for k, v in ret.items()]))
    # ret = get_halt_security_ids(security_ids=['100567.XSHG', '100096.XSHG'], curr_date='20050614')

    # ret = get_idx_weights(security_id='000300.XSHG', index_date='20190618', source=0)
    # print(ret)

    # ret = get_sw_2nd_indust(security_ids=['001979.XSHE', '603612.XSHG'], source=0)
    # print(ret)

    # ret1 = get_idx_cons('000300.XSHG', "20190618")
    # ret2 = get_idx_weights('000300.XSHG', "20190618")
    # print(len(ret1))
    # print(len(ret2))
    # print(set(ret1) - set(ret2.keys()))
    # print(set(ret2.keys()) - set(ret1))

    # ret = get_idx_adjust_dates('000300.XSHG', '20150103', '20190620')
    # print(ret)

    # ret = get_security_codes()
    # print(ret)

    # ret = get_sw_indust()
    # print(ret)

    # ret = get_idx_cons_jy('000300', '20190801', '20190826')
    # pprint.pprint(ret)

    # ret = get_idx_adjust_dates_jy(idx_security_id='00300', start_date='20190103', end_date='20190826', source=0)
    # pprint.pprint(ret)

    # ret = get_indust_mkt(start_date='20190701', end_date='20190826')
    # print('*'*40)
    # pprint.pprint(ret)

    ret = get_sw_2nd_indust(security_ids=['001979.XSHE'])
    print(ret)
