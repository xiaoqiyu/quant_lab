# -*- coding: utf-8 -*-
# @time      : 2019/5/20 10:39
# @author    : rpyxqi@gmail.com
# @file      : feature_research.py


import pprint
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

from quant_models.data_processing.features_calculation import get_equity_daily_features
from quant_models.data_processing.features_calculation import get_idx_adjust_dates_jy
from quant_models.data_processing.features_calculation import get_idx_cons_jy
from quant_models.data_processing.features_calculation import get_idx_returns
from quant_models.data_processing.features_calculation import get_equity_returns
from quant_models.data_processing.features_calculation import get_source_feature_mappings
from quant_models.data_processing.features_calculation import get_market_value
from quant_models.data_processing.features_calculation import get_sw_indust
from quant_models.data_processing.features_calculation import get_indust_mkt
from quant_models.model_processing.feature_preprocessing import feature_preprocessing
from quant_models.data_processing.features_calculation import get_security_codes
from quant_models.data_processing.features_calculation import get_sw_2nd_indust
from quant_models.model_processing.ml_reg_models import Ml_Reg_Model
from quant_models.utils.date_utils import datetime_delta
from quant_models.utils.io_utils import write_json_file
from quant_models.utils.logger import Logger
from quant_models.utils.helper import get_config
from quant_models.utils.helper import get_source_root
from scipy.optimize import lsq_linear

logger = Logger(log_level='DEBUG', handler='ch').get_log()
config = get_config()
feature_model_name = 'random_forest'


def _get_source_features():
    source_features = config['source_features']
    ret = {}
    for k, v in source_features.items():
        ret.update({k: v.split(',')})
    return ret


def feature_preprocessing(arr=None, fill_none=False, weights=None):
    if isinstance(arr, list):
        arr = np.array(arr)
    if fill_none:
        _input_pip = Pipeline([
            ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
        ])
        try:
            arr = _input_pip.fit_transform(arr)
        except Exception as ex:
            logger.info('fail to apply the imputer with error:{0}'.format(ex))
    n_shape = arr.shape
    try:
        if len(n_shape) > 1:
            n_col = n_shape[1]
        else:
            n_col = len(arr[0])
    except Exception as ex:
        logger.error('input array size not match:{0}'.format(ex))
        return
    _ret = []
    for idx in range(n_col):
        try:
            _tmp = [item[idx] for item in arr]
            _tmp = win_and_std(_tmp, weights)
            _ret.append(_tmp)
        except Exception as ex:
            continue
    return np.array(_ret).transpose()


def win_and_std(arr=None, weights=[]):
    w_mean = (np.array(arr) * np.array(weights)).mean()
    mean, std = np.array(arr).mean(), np.array(arr).std()
    _min = mean - 3 * std
    _max = mean + 3 * std
    arr = np.clip(arr, a_min=_min, a_max=_max)
    return arr if std == 0 else [(item - w_mean) / std for item in arr]


def get_sw_2nd_vectors(security_ids=[]):
    sw_2nd_dict = get_sw_2nd_indust(security_ids=security_ids)
    all_names = list(set(sw_2nd_dict.values()))
    ret_exposures = []
    for k, v in sw_2nd_dict.items():
        _v = [1 if v == item else 0 for item in all_names]
        ret_exposures.append(_v)
    return ret_exposures, all_names


# FIXME add the industry update information
def get_indust_vectors(security_ids=[]):
    '''
    use the binary encode for the industry vector, however, did not add the update info yet
    :param security_ids:
    :return:
    '''
    indus_mapping = get_sw_indust()
    indus_lst = [indus_mapping.get(k) for k in security_ids]
    indus_set = list(set(indus_lst))

    try:
        indus_set.remove(None)
    except Exception as ex:
        pass
    try:
        indus_set.remove(np.nan)
    except Exception as ex:
        pass
    ret = []
    n_indus = len(indus_set)
    for idx, val in enumerate(indus_lst):
        try:
            # TODO handle the case when val is None
            ret.append([1 if val and _idx == indus_set.index(val) else 0 for _idx in range(n_indus)])
        except Exception as ex:
            logger.error(ex)
    return np.array(ret)


def get_indust_exposures(start_date=None, end_date=None, stock_returns=None):
    logger.info("Start processing the industry exposure for start_date:{0}, end_date:{1} ".format(start_date, end_date))
    df_indust = get_indust_mkt(start_date=start_date, end_date=end_date)
    indust_codes = list(set(df_indust['IndustryCode']))
    indust_names = list(set(df_indust['IndustryName']))
    indust_rows = []
    for item in indust_codes:
        indust_rows.append(list(df_indust[df_indust.IndustryCode == item].sort_values(by='TradingDay')['ChangePCT']))
    x = pd.DataFrame(indust_rows).transpose()
    # FIXME check the rows number for x
    # x = x[list(range(16))]
    # del df_indust
    exposure_rows = []

    for sec_id, return_dict in stock_returns.items():
        return_rows = list(zip(list(return_dict.keys()), list(return_dict.values())))
        return_rows = sorted(return_rows, key=lambda x: x[0])
        y = [item[1] for item in return_rows]
        # y = y - _last_col
        # y = [_y[idx] - val for idx, val in enumerate(_last_col)]
        n_col = x.shape[1]
        # assert x.shape[0] == len(y)
        x = x[-x.shape[1]:]
        y = y[-x.shape[1]:]
        x = x.append([[1.0] * x.shape[1]])
        y.append(1.0)

        m = Ml_Reg_Model('linear_nnls')
        m.build_model()
        m.train_model(x, y, sample_weight=(0, 1.0))
        coef, intercept = m.output_model()
        exposure_rows.append(coef)
    logger.info(
        "Complete processing the industry exposure for start_date:{0}, end_date:{1} with return rows:{2} ".format(
            start_date, end_date, len(exposure_rows)))
    return exposure_rows, indust_names


def _get_in_out_dates(start_date=None, end_date=None, security_id=None):
    _dates = get_idx_adjust_dates_jy(idx_security_id=security_id, start_date=start_date, end_date=end_date, source=0)
    _dates.append(start_date)
    _dates.append(end_date)
    _dates = list(set(_dates))
    _dates.sort()
    ret = []
    for d in _dates[:-1]:
        _idx = _dates.index(d)
        ret.append([_dates[_idx], _dates[_idx + 1]])
    return ret


def factor_return_regression(country_factors, indus_factors, deriv_factors, returns):
    logger.info("Start final regression for all factors")
    m = Ml_Reg_Model('linear')
    m.build_model()
    x = np.hstack((country_factors, indus_factors, deriv_factors))
    m.train_model(x, returns)
    coef, intercept = m.output_model()
    logger.info("Complete final regression for all factors with coef:{0} and intercept:{1}".format(coef, intercept))
    return coef


def get_factor_returns(start_date='20181101', end_date='20181131', data_source=0,
                       feature_types=[], saved_feature=True, bc=None,
                       top_ratio=0.25, bottom_ratio=0.2):
    root = get_source_root()
    feature_mapping = _get_source_features() or get_source_feature_mappings(souce=True, feature_types=feature_types,
                                                                            top_ratio=top_ratio,
                                                                            bottom_ratio=bottom_ratio)
    date_periods = _get_in_out_dates(start_date=start_date, end_date=end_date, security_id='000300.XSHG') or [
        [start_date, end_date]]
    next_date = datetime_delta(dt=end_date, format='%Y%m%d', days=1)
    idx_labels = get_idx_returns(security_ids=[bc], start_date=start_date, end_date=next_date,
                                 source=0).get(bc)
    all_factor_returns = []

    for _start_date, _end_date in date_periods:
        logger.info('get factor return: processing from {0} to {1}'.format(_start_date, _end_date))
        next_date = datetime_delta(dt=_end_date, format='%Y%m%d', days=2)
        if bc:
            security_ids = get_idx_cons_jy(bc, _start_date, _end_date)
        else:
            security_ids = get_security_codes()
        # FIXME HACK FOR TESTING
        security_ids = security_ids[:5]
        ret_features = get_equity_daily_features(security_ids=security_ids, features=feature_mapping,
                                                 start_date=_start_date,
                                                 end_date=_end_date, source=data_source)
        ret_returns = get_equity_returns(security_ids=security_ids, start_date=_start_date, end_date=next_date)
        ret_mv = get_market_value(security_ids=security_ids, start_date=_start_date, end_date=next_date)
        none_factor_dict = defaultdict()
        # FIXME use the future industry return, should be updated to trace back the history data by windows
        industry_exposure_factors = get_indust_exposures(start_date=_start_date, end_date=_end_date,
                                                         stock_returns=ret_returns)
        _industry_exposure_df = pd.DataFrame(industry_exposure_factors)
        _industry_exposure_df.to_csv('{0}_{1}_{2}.csv'.format(_start_date, _end_date, bc))
        for date, val in ret_features.items():
            daily_factors = []
            daily_return = []
            daily_mv = []
            for sec_id, f_lst in val.items():
                _val_lst = list(f_lst.values())
                all_feature_names = list(f_lst.keys())
                daily_factors.append(_val_lst)
                try:
                    s_label = ret_returns.get(sec_id).get(str(date))
                    i_label = idx_labels.get(str(date))
                    label = (s_label - i_label) * 100
                    mv = ret_mv.get(sec_id).get(str(date))
                except Exception as ex:
                    label = np.nan
                    logger.error('fail to calculate the label with error:{0}'.format(ex))
                daily_return.append(label)
                daily_mv.append(mv)
            try:
                daily_factors = feature_preprocessing(arr=daily_factors, fill_none=True, weights=daily_mv)
            except Exception as ex:
                logger.error('fail in feature preprocessing with error:{0}'.format(ex))
            # indus_factors = get_indust_vectors(security_ids)
            courtry_factors = np.ones(len(security_ids)).reshape(len(security_ids), 1)
            factor_returns = factor_return_regression(courtry_factors, industry_exposure_factors, daily_factors,
                                                      daily_return)
            all_factor_returns.append(factor_returns)
    mean_returns = np.array(all_factor_returns).mean(axis=0)
    return mean_returns


if __name__ == '__main__':
    # ret = get_factor_returns(start_date='20190801', end_date='20190805', data_source=0, feature_types=[],
    #                          bc='000300.XSHG')
    # pprint.pprint(ret)
    # ret = get_indust_exposures(start_date='20190103', end_date='20190706')
    # pprint.pprint(ret)
    # ret = _get_source_features()
    # pprint.pprint(ret)

    # testing code
    # secs = ['001979.XSHE', '603612.XSHG']
    # start_date = '20190616'
    # end_date = '20190923'
    # ret_returns = get_equity_returns(security_ids=secs, start_date=start_date, end_date=end_date)
    # industry_exposure_factors, industry_name = get_indust_exposures(start_date=start_date, end_date=end_date,
    #                                                                 stock_returns=ret_returns)
    # # df = pd.DataFrame(industry_exposure_factors, columns=industry_name, index=secs)
    # pprint.pprint(industry_name)
    # pprint.pprint(list(industry_exposure_factors[0]))

    # test the multi-factor
    vec, names = get_sw_2nd_vectors(['001979.XSHE', '603612.XSHG'])
    print(names)
    print(vec)
