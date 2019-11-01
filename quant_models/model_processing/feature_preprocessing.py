# -*- coding: utf-8 -*-
# @time      : 2018/12/11 13:43
# @author    : rpyxqi@gmail.com
# @file      : feature_preprocessing.py

import os
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
from scipy import stats
from minepy import MINE
from quant_models.utils.logger import Logger
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from quant_models.utils.helper import get_source_root
from quant_models.model_processing.ml_reg_models import Ml_Reg_Model

logger = Logger(log_level='INFO', handler='ch').get_log()


# reference:
# http://www.cnblogs.com/stevenlk/p/6543628.html
# https://scikit-learn.org/stable/modules/feature_selection.html

def feature_selection(x=[], y=[], feature_names=[], sort_type='pearson', reverse=True, topk=10):
    ret = feature_selection_sort(x, y, feature_names, sort_type, reverse)
    return ret[:topk]


def feature_selection_sort(x=[], y=[], feature_names=[], sort_type='pearson', reverse=True):
    '''
    :param x:
    :param y:
    :param corr_type: 'person'|'spearman'|'mic'|'rf'|'dcor'
    :return:
    '''
    # TODO to add dcor
    if sort_type in ['pearson', 'spearman']:
        scores = _sort_feature_by_corr(x, y, feature_names, sort_type)
    else:
        scores = {
            'rf': _sort_feature_by_rf(x, y, feature_names),
            'mic': _sort_feature_by_mic(x, y, feature_names),
        }
    scores.sort(key=lambda x: x[0], reverse=reverse)
    return scores


def _sort_feature_by_corr(x=[], y=[], feature_names=[], corr_type='pearson'):
    df_corr = pd.DataFrame(x)
    df_corr[df_corr.shape[1]] = y
    corr = df_corr.corr(corr_type)
    corr_lst = corr.iloc[-1][:-1]
    scores = []
    for idx, item in enumerate(corr_lst):
        scores.append((item, feature_names[idx]))
    return scores


def _sort_feature_by_mic(x=[], y=[], feature_names=[]):
    '''

    I(X;Y)=E[I(xi;yj)]=∑xiϵX∑yjϵYp(xi,yj)logp(xi,yj)p(xi)p(yj)
    :param x:
    :param y:
    :param feature_names:
    :return:
    '''
    m = MINE()
    x = x.T
    scores = []
    for idx, item in enumerate(x):
        m.compute_score(item, y)
        scores.append((m.mic(), feature_names[idx]))
    return scores


def _sort_feature_by_rf(x=[], y=[], feature_names=[]):
    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = []
    for i in range(x.shape[1]):
        score = cross_val_score(rf, x[:, i:i + 1], y, scoring='r2', cv=ShuffleSplit(len(x), 3, .3))
        scores.append((np.mean(score), feature_names[i]))
    return scores


def feature_selection_wrapper(x=[], y=[]):
    pass


def feature_selection_embedded(x=[], y=[]):
    lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(x, y)
    model = SelectFromModel(lsvc, prefit=True)
    return model.transform(x)


def get_sw1_indust_code(sec_ids=[], trade_date=''):
    root = get_source_root()
    indust_path = os.path.join(os.path.join(root, 'data'), 'features', 'sw1_indust.csv')
    df = pd.read_csv(indust_path)
    ret = {}
    for sec_id in sec_ids:
        _df = df[df.secID == sec_id]
        for idx, sec_id, in_code, into_date, out_date, is_new in list(_df.values):
            if not is_new:
                if into_date.replace('-', '') <= str(trade_date) and str(trade_date) <= out_date.replace('-', ''):
                    ret.update({sec_id: in_code})
                    continue
            else:
                if into_date.replace('-', '') <= str(trade_date):
                    ret.update({sec_id: in_code})
    return ret


def feature_preprocessing(arr=None, fill_none=False, trade_date='', sec_ids=[], neutralized=False):
    logger.info("Start feature_prepcessing...")
    if isinstance(arr, list):
        arr = np.array(arr)

    if fill_none:
        _input_pip = Pipeline([
            ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
        ])
        try:
            arr = _input_pip.fit_transform(arr)
        except Exception as ex:
            logger.error('fail to apply the imputer with error:{0}'.format(ex))
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
            _tmp = win_and_std(_tmp)
            _ret.append(_tmp)
        except Exception as ex:
            continue
    _ret = np.array(_ret).transpose()
    logger.info("Complete feature_prepcessing")
    if not neutralized:
        return _ret
    # FIXME check the industry neutralized logic
    indust_ret = get_sw1_indust_code(sec_ids, trade_date)
    indust_codes = list(indust_ret.values())
    all_indust_set = list(set(indust_codes))
    indust_features = np.array([[1 if _tmp == item else 0 for _tmp in all_indust_set] for item in indust_codes])
    m = Ml_Reg_Model('linear')
    m.build_model()
    m.train_model(indust_features, _ret)
    pred_y = m.predict(indust_features)
    logger.info("Complete feature_prepcessing")
    return _ret - pred_y


def win_and_std(arr=None):
    mean, std = np.array(arr).mean(), np.array(arr).std()
    _min = mean - 3 * std
    _max = mean + 3 * std
    arr = np.clip(arr, a_min=_min, a_max=_max)
    return arr if std == 0 else [(item - mean) / std for item in arr]

# if __name__ == '__main__':
# linear_reg()
# b = load_boston()
# x = b['data']
# y = b['target']
# names = b['feature_names']
# ret = feature_selection_sort(x, y, names)
# print(ret)

# iris = load_iris()
# x, y = iris.data, iris.target
# x_new = feature_selection_embedded(x, y)
# print(x_new[:10])

# x = np.random.random(10).reshape(5, 2)
# x[0, 0] = 10
# x1 = feature_preprocessing(x)
# print(x1)

# df = get_sw1_indust_code(['000001.XSHE', '000002.XSHE'], '20190531')
# print(df)
