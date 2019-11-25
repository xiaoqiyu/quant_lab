# -*- coding: utf-8 -*-
# @time      : 2019/11/21 20:04
# @author    : rpyxqi@gmail.com
# @file      : hf_feature_selection.py

from quant_models.utils.logger import Logger
from quant_models.utils.helper import get_config
from quant_models.data_processing.features_calculation import get_idx_adjust_dates
from quant_models.data_processing.features_calculation import get_idx_cons_dy
from quant_models.data_processing.features_calculation import get_equity_returns
from quant_models.model_processing.feature_preprocessing import feature_preprocessing
from quant_models.data_processing.features_calculation import get_idx_returns
from quant_models.data_processing.hf_features_calculation import Market
from quant_models.data_processing.hf_features_calculation import get_min_features
import numpy as np
import pandas as pd
from WindPy import w
import gc

logger = Logger(log_level='DEBUG', handler='ch').get_log()
config = get_config()
w.start()

hf_features = ['ret_var', 'ret_skew', 'ret_kurtosis']


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


def _next_trading_date(tdays=[], trade_date=''):
    # assume tdays is sorted and has enough dates
    cnt = 0
    tdays = sorted(tdays)
    for d in tdays:
        if d < trade_date:
            cnt += 1
    return tdays[cnt]


def retrieve_features(start_date='20181101', end_date='20181131', data_source=0,
                      feature_types=[], bc='000300.XSHG'):
    # feature_mapping = get_source_feature_mappings(feature_types=feature_types)
    date_periods = _get_in_out_dates(start_date=start_date, end_date=end_date, security_id='000300.XSHG') or [
        [start_date, end_date]]
    all_labels = []
    all_features = []
    all_feature_names = []
    g_next_date = w.tdaysoffset(1, end_date).Data[0][0].strftime('%Y%m%d')
    idx_labels = get_idx_returns(security_ids=[bc], start_date=start_date, end_date=g_next_date,
                                 source=0).get(bc)
    _w_ret = w.tdays(start_date, g_next_date)
    tdays = [item.strftime(config['constants']['standard_date_format']) for item in _w_ret.Data[0]]
    tdays = sorted(tdays)
    for _start_date, _end_date in date_periods:
        next_date = _next_trading_date(tdays, _end_date)
        if not next_date:
            logger.error("Fail to get the next trading date for :{0}".format(_end_date))
        security_ids = get_idx_cons_dy(bc, _start_date)
        # FIXME add some filter,e.g. halt sec
        logger.info("Start query the features from :{0} to {1}....".format(_start_date, _end_date))
        mkt = Market(security_ids)
        # mkt.security_ids = ['000651.XSHE']
        mkt.initialize(start_date=_start_date, end_date=next_date, source=0, tick_cache='')
        ret_features = get_min_features(security_ids=security_ids, start_date=_start_date, end_date=_end_date,
                                        start_time=None,
                                        end_time=None, mkt_cache=mkt, features=hf_features)

        # ret_features = get_equity_daily_features(security_ids=security_ids, features=feature_mapping,
        #                                          start_date=_start_date,
        #                                          end_date=_end_date, source=data_source)
        logger.info("Complete query the features from :{0} to {1}".format(_start_date, _end_date))
        logger.info("Start query the market returns from :{0} to {1}....".format(_start_date, next_date))
        ret_labels = get_equity_returns(security_ids=security_ids, start_date=_start_date, end_date=next_date)
        logger.info("Complete query the market returns from :{0} to {1}".format(_start_date, next_date))
        logger.info("Start calculate the features from :{0} to {1}....".format(_start_date, _end_date))
        for date, val in ret_features.items():
            logger.info("Starting processing for date:{0}....".format(date))
            date_features = []
            date_labels = []
            for sec_id, f_lst in val.items():
                _val_lst = list(f_lst.values())
                all_feature_names = list(f_lst.keys())
                date_features.append(_val_lst)
                try:
                    _next_date = _next_trading_date(tdays, str(date))
                    s_label = ret_labels.get(sec_id).get(str(_next_date))
                    i_label = idx_labels.get(str(_next_date))
                    # label = s_label * 100 - i_label
                    label = s_label * 100
                except Exception as ex:
                    label = np.nan
                    logger.error('Fail to calculate the label with error:{0}'.format(ex))
                date_labels.append(label)

            try:
                date_features = feature_preprocessing(arr=date_features, fill_none=True, trade_date=date,
                                                      sec_ids=list(val.keys()), neutralized=False)
            except Exception as ex:
                logger.error('Fail in feature preprocessing with error:{0}'.format(ex))
            logger.info('Adding sec_id and trade_dates and reshape...')
            try:
                df_shape = date_features.shape
                # df_shape = [len(date_features), len(date_features[0])]
                date_features = np.column_stack((date_features, list(val.keys()), [date] * df_shape[0]))
                all_features.extend(date_features)
                all_labels.extend(date_labels)
            except Exception as ex:
                logger.error('fail to reshape features features with error:{0}'.format(ex))
            logger.info("Complete processing for date:{0}".format(date))
        logger.info("Complete calculate the features from :{0} to {1}".format(_start_date, _end_date))
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
    df.to_csv('hf_feature_6m_abs.csv')
    corr_var = np.corrcoef([float(item) for item in list(df['ret_var'])], df['LABEL'])
    corr_skew = np.corrcoef([float(item) for item in list(df['ret_skew'])], df['LABEL'])
    corr_kurtosis = np.corrcoef([float(item) for item in list(df['ret_kurtosis'])], df['LABEL'])
    print(df.corr())
    return df, corr_var, corr_skew, corr_kurtosis


if __name__ == '__main__':
    df, corr_var, corr_skew, corr_kurtosis = retrieve_features(start_date='20190103', end_date='20190628', data_source=0,
                      feature_types=[], bc='000300.XSHG')
    print(corr_var)
    print(corr_skew)
    print(corr_kurtosis)

