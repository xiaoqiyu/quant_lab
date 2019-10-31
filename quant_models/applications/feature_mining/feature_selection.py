# -*- coding: utf-8 -*-
# @time      : 2019/5/20 10:39
# @author    : rpyxqi@gmail.com
# @file      : feature_selection.py


from collections import OrderedDict
import numpy as np
import pprint
from collections import defaultdict
import time
import os
import gc
from WindPy import w
import pandas as pd
from quant_models.data_processing.features_calculation import get_equity_daily_features
from quant_models.data_processing.features_calculation import get_equity_returns
from quant_models.data_processing.features_calculation import get_idx_returns
from quant_models.data_processing.features_calculation import get_source_feature_mappings
from quant_models.data_processing.features_calculation import get_idx_cons_dy
from quant_models.data_processing.features_calculation import get_idx_adjust_dates
from quant_models.data_processing.features_calculation import save_features
from quant_models.data_processing.features_calculation import read_features
from quant_models.utils.logger import Logger
from quant_models.utils.helper import get_config
from quant_models.utils.io_utils import write_json_file
from quant_models.utils.io_utils import load_json_file
from quant_models.utils.helper import get_source_root
from quant_models.utils.date_utils import datetime_delta
from quant_models.model_processing.feature_preprocessing import feature_preprocessing
from quant_models.utils.decorators import timeit

logger = Logger(log_level='DEBUG', handler='ch').get_log()
config = get_config()
feature_model_name = 'random_forest'
w.start()


def _resolve_none(arr=[], col_names=[], none_dict={}):
    # np.array()
    sum_dict = defaultdict(list)
    # none_dict = defaultdict()
    mean_dict = defaultdict()
    for item in arr:
        for idx, val in enumerate(item):
            if val:
                sum_dict[col_names[idx]].append(val)
            else:
                _tmp = none_dict.get(col_names[idx]) or 0
                _tmp += 1
                none_dict.update({col_names[idx]: _tmp})
    for k, v in sum_dict.items():
        mean_dict.update({k: sum(v) / len(v)})
    for item in arr:
        for idx, val in enumerate(item):
            if not col_names[idx] in ['SECURITY_ID', 'TRADE_DATE']:
                if not val or val != val:
                    item[idx] = mean_dict[col_names[idx]]
    return arr


def _get_in_out_dates(start_date=None, end_date=None, security_id=None):
    _dates = get_idx_adjust_dates(security_id=security_id, start_date=start_date, end_date=end_date, source=0)
    _dates.append(start_date)
    _dates.append(end_date)
    _dates = list(set(_dates))
    _dates.sort()
    ret = []
    for d in _dates[:-1]:
        _idx = _dates.index(d)
        ret.append([_dates[_idx], _dates[_idx + 1]])
    return ret


def score_json2csv():
    root = get_source_root()
    score_path = os.path.join(os.path.realpath(root), 'conf',
                              '20190605_testing_train_features_score_20150103_20181231.json')
    corr_dict = load_json_file(score_path)
    feature_mapping = get_source_feature_mappings(train_feature=True)
    rows = []
    for k, v in corr_dict.items():
        if k == 'TCOSTTTM':
            print('testing')
        for ft, lst in feature_mapping.items():
            if k in lst:
                rows.append([k, v, ft])
                continue
    pprint.pprint(rows)
    df = pd.DataFrame(rows, columns=['key', 'score', 'feature_type'])
    score_path = os.path.join(os.path.realpath(root), 'data', 'features',
                              'score_20150103_20181231.csv')
    df.to_csv(score_path)


def retrieve_features(start_date='20181101', end_date='20181131', data_source=0,
                      feature_types=[], bc='000300.XSHG'):
    feature_mapping = get_source_feature_mappings(feature_types=feature_types)
    date_periods = _get_in_out_dates(start_date=start_date, end_date=end_date, security_id='000300.XSHG') or [
        [start_date, end_date]]
    all_labels = []
    all_features = []
    all_feature_names = []
    g_next_date = datetime_delta(dt=end_date, format='%Y%m%d', days=1)
    idx_labels = get_idx_returns(security_ids=[bc], start_date=start_date, end_date=g_next_date,
                                 source=0).get(bc)
    for _start_date, _end_date in date_periods:
        next_date = datetime_delta(dt=_end_date, format='%Y%m%d', days=2)
        security_ids = get_idx_cons_dy(bc, _start_date)
        # FIXME add some filter,e.g. halt sec
        ret_features = get_equity_daily_features(security_ids=security_ids, features=feature_mapping,
                                                 start_date=_start_date,
                                                 end_date=_end_date, source=data_source)
        ret_labels = get_equity_returns(security_ids=security_ids, start_date=_start_date, end_date=next_date)
        for date, val in ret_features.items():
            date_features = []
            date_labels = []
            for sec_id, f_lst in val.items():
                _val_lst = list(f_lst.values())
                all_feature_names = list(f_lst.keys())
                date_features.append(_val_lst)
                try:
                    s_label = ret_labels.get(sec_id).get(str(date))
                    i_label = idx_labels.get(str(date))
                    label = (s_label - i_label) * 100
                except Exception as ex:
                    label = np.nan
                    logger.error('fail to calculate the label with error:{0}'.format(ex))
                date_labels.append(label)
            try:
                date_features = feature_preprocessing(arr=date_features, fill_none=True, trade_date=date,
                                                      sec_ids=list(val.keys()), neutralized=False)
            except Exception as ex:
                logger.error('fail in feature preprocessing with error:{0}'.format(ex))
            try:
                df_shape = date_features.shape
                date_features = np.column_stack((date_features, list(val.keys()), [date] * df_shape[0]))
                all_features.extend(date_features)
                all_labels.extend(date_labels)
            except Exception as ex:
                logger.error('fail to reshape features features with error:{0}'.format(ex))
    try:
        if 'SECURITY_ID' not in all_feature_names:
            all_feature_names.append('SECURITY_ID')
        if 'TRADE_DATE' not in all_feature_names:
            all_feature_names.append('TRADE_DATE')
        df = pd.DataFrame(all_features, columns=all_feature_names)
        df['LABEL'] = all_labels
    except Exception as ex:
        logger.error(ex)
    del ret_features
    del ret_labels
    gc.collect()
    return df


@timeit
def cache_features(start_date='20180101', end_date='20181231', data_source=0,
                   feature_types=[], bc='000300.XSHG'):
    '''
    cache the features
    :param start_date:
    :param end_date:
    :param data_source:
    :param feature_types:
    :param bc:
    :return:
    '''
    df = retrieve_features(start_date=start_date, end_date=end_date, data_source=data_source,
                           feature_types=feature_types, bc=bc)
    # save_features(tuple(df.columns), tuple(df.values))
    root = get_source_root()

    # df.to_pickle(feature_source)
    df['MONTH'] = [item[:6] for item in df['TRADE_DATE']]
    m_dates = set(df['MONTH'])
    n_mdates = len(m_dates)
    for idx, m_date in enumerate(m_dates):
        _df = df[df.MONTH == m_date]
        feature_source = os.path.join(os.path.realpath(root), 'data', 'features',
                                      'features{0}_{1}.csv'.format(bc.split('.')[0], m_date))
        logger.info('Saving the {0} th features {1} out of {2}'.format(idx, feature_source, n_mdates))
        _df.to_csv(feature_source, index=None)
    del df
    gc.collect()


@timeit
def train_features(start_date='', end_date='', bc='000300.XSHG'):
    # rows, desc = read_features()
    # _df = pd.DataFrame(rows, columns=desc)
    _w_ret = w.tdays(start_date, end_date)
    t_months = list(set([item.strftime('%Y%m') for item in _w_ret.Data[0]]))
    root = get_source_root()
    feature_paths = [os.path.join(os.path.realpath(root), 'data', 'features',
                                  'features{0}_{1}.csv'.format(bc.split('.')[0], m_date)) for m_date in t_months]
    if feature_paths:
        df = pd.read_csv(feature_paths[0])
        for p in feature_paths[1:]:
            df.append(pd.read_csv(p))
    cols = list(df.columns)[:-4]
    cols.append('LABEL')
    df_corr = df[cols].corr()
    score_df = pd.DataFrame({'feature': cols[:-1], 'score': df_corr['LABEL'][:-1]})
    score_path = os.path.join(os.path.realpath(root), 'data', 'features',
                              'score_{0}_{1}.csv'.format(start_date, end_date))
    score_df.to_csv(score_path, index=None)
    return score_df


if __name__ == '__main__':
    import gc
    cache_features(start_date='20170103', end_date='20171231', data_source=0,
                   feature_types=[], bc='000300.XSHG')
    gc.collect()
    cache_features(start_date='20180103', end_date='20181231', data_source=0,
                   feature_types=[], bc='000300.XSHG')
    gc.collect()
    cache_features(start_date='20190103', end_date='20190531', data_source=0,
                   feature_types=[], bc='000300.XSHG')
    gc.collect()
    # train_features(start_date='20190103', end_date='20190531', bc='000300.XSHG')
