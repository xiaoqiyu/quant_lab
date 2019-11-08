# -*- coding: utf-8 -*-
# @time      : 2019/5/22 15:12
# @author    : rpyxqi@gmail.com
# @file      : model_selection.py

import time
import os
import gc
import pandas as pd
from quant_models.model_processing.ml_reg_models import Ml_Reg_Model
from quant_models.utils.logger import Logger
from quant_models.utils.helper import get_config
from quant_models.utils.helper import get_source_root
from quant_models.utils.decorators import timeit
from quant_models.applications.feature_mining.feature_selection import train_features
from sklearn import decomposition
from sklearn.ensemble import gradient_boosting
import datetime

logger = Logger(log_level='DEBUG', handler='ch').get_log()
config = get_config()
feature_model_name = 'random_forest'


def get_selected_features():
    root = get_source_root()
    feature_source = os.path.join(os.path.realpath(root), 'data', 'features')
    # files = os.listdir(feature_source)
    # files = [item for item in files if item.startswith('score')]
    # get the score with the corresponding start and end date, otherwise return the latest one
    # TODO confirm whether the listdir function's return is sorted by time
    # _path = 'score_{0}_{1}'.format(start_date, end_date) if 'score_{0}_{1}'.format(start_date, end_date) in files else \
    #     files[-1]
    # _path = files[0]
    _score_path = os.path.join(feature_source,
                               'score_{0}_{1}.csv'.format(config['feature_mining_strategy']['start_date'],
                                                          config['feature_mining_strategy']['end_date']))
    # logger.info("Reading score from path:{0}".format(_score_path))
    score_df = pd.read_csv(_score_path)
    score_df = score_df.sort_values(by='score', ascending=False)
    _feature_names = list(score_df['feature'])
    _score_bound = int(len(_feature_names) * float(config['feature_mining_strategy']['best_feature_ratio']) / 2)
    feature_names = list(
        set(_feature_names[:_score_bound + 1]).union(set(_feature_names[-_score_bound:])))
    return feature_names


@timeit
def train_models(model_name='', start_date='20140603', end_date='20181231', feature_ratio=1.0, bc='000300.XSHG',
                 feature_df=None, score_df=None, cache_df=False):
    '''

    :param model_name:
    :param start_date:
    :param end_date:
    :param score_bound: (up_ratio, down_ratio), will pick the features with socre in the top up_ratio ranking
    and in the bottom with down_ratio
    :return:
    '''
    m = Ml_Reg_Model(model_name)
    model_full_name = '{0}_{1}_{2}_{3}'.format(model_name, start_date, end_date, feature_ratio, bc)
    if not m.load_model(model_full_name):
        m.build_model()
    root = get_source_root()
    if not cache_df:
        df, score_df = train_features(start_date=start_date, end_date=end_date, bc=bc)
    else:
        df = feature_df
        score_df = score_df
    score_df = score_df.sort_values(by='score', ascending=False)
    _feature_names = list(score_df['feature'])
    _score_bound = int(len(_feature_names) * feature_ratio / 2)
    feature_names = list(
        set(_feature_names[:_score_bound + 1]).union(set(_feature_names[-_score_bound:])))

    # select the features by the ic values
    # feature_names = get_selected_features(start_date=start_date, end_date=end_date, up_ratio=score_bound[0],
    #                                       down_ratio=score_bound[1])

    train_X = df[feature_names].values
    dataes = list(df['TRADE_DATE'])
    sec_ids = list(df['SECURITY_ID'])
    train_Y = df.iloc[:, -1]
    decom_ratio = float(config['feature_mining_strategy']['component_ratio'])
    # PCA processing
    n_component = min(int(len(feature_names) * decom_ratio), int(config['feature_mining_strategy']['n_component']))
    pca = decomposition.PCA(n_components=n_component)
    # train_X = pca.fit_transform(train_X)
    train_X = pca.fit_transform(pd.DataFrame(train_X).fillna(method='ffill'))
    train_Y = train_Y.fillna(0.0)
    st = time.time()
    logger.info('start training the models')
    # mse_scores, r2_scores = m.train_model(train_X[:1000], train_Y[:1000])
    mse_scores, r2_scores = m.train_model(train_X, train_Y)
    et = time.time()
    logger.info('complete training model with time:{0}'.format(et - st))

    m.save_model(model_full_name)
    logger.info("Mean squared error:{0}: %0.5f -  %0.5f" % (
        mse_scores.mean() - mse_scores.std() * 3, mse_scores.mean() + mse_scores.std() * 3))
    logger.info("Mean squared error:{0}: %0.5f -  %0.5f" % (
        r2_scores.mean() - r2_scores.std() * 3, r2_scores.mean() + r2_scores.std() * 3))
    result_path = os.path.join(os.path.join((os.path.join(os.path.realpath(root), 'data')), 'results'),
                               'feature_model_selection.txt')

    with open(result_path, 'a+') as fout:
        # fout.write('{0}\n'.format(datetime.datetime.now().strftime(config['constants']['no_dash_datetime_format'])))
        fout.write('{0}\n'.format(model_full_name))
        fout.write('mse: {0}\n r2_score:{1}\n'.format(str(list(mse_scores)), str(list(r2_scores))))
    return mse_scores, r2_scores


def train_model_selections():
    model_names = config['feature_mining_strategy']['model_names'].split(',')
    bc = config['feature_mining_strategy']['benchmark']
    start_date = config['feature_mining_strategy']['start_date']
    end_date = config['feature_mining_strategy']['end_date']
    _feature_ratios = config['feature_mining_strategy']['feature_ratios'].split(',')
    feature_ratios = [float(item) for item in list(_feature_ratios)]
    best_mse = 0.0
    best_score = 0.0
    best_model = ''
    cv = int(config['ml_reg_model']['cv'])
    df, score_df = train_features(start_date=start_date, end_date=end_date, bc=bc)
    for model_name in model_names:
        for f_ratio in feature_ratios:
            # FIXME grid search for some specific model
            mse, r2 = train_models(model_name=model_name, start_date=start_date, end_date=end_date,
                                   feature_ratio=f_ratio,
                                   bc=bc, feature_df=df, score_df=score_df, cache_df=True)
            logger.info('Train Results for {0},{1},{2} are mes:{3},r2_score:{4}'.format(model_name, f_ratio, bc, mse,
                                                                                        r2))
            if sum(list(mse)) / cv > best_mse:
                best_model = '{0}_{1}'.format(model_name, f_ratio)
                best_mse = sum(list(mse)) / cv
            if sum(list(r2)) / cv > best_score:
                best_score = sum(list(r2)) / cv
    print(best_mse, best_score, best_model)
    del df
    del score_df
    gc.collect()
    return best_mse, best_score, best_model


if __name__ == '__main__':
    st = time.time()
    # ret = train_models(model_name='linear', start_date='20170103', end_date='20181231', )
    ret = train_model_selections()
    et = time.time()
    print(et - st)
    print(ret)
