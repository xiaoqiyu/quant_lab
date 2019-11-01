# -*- coding: utf-8 -*-
# @time      : 2019/5/22 15:12
# @author    : rpyxqi@gmail.com
# @file      : model_selection.py

import time
import os
import pandas as pd
from quant_models.model_processing.ml_reg_models import Ml_Reg_Model
from quant_models.utils.logger import Logger
from quant_models.utils.helper import get_config
from quant_models.utils.helper import get_source_root
from quant_models.utils.decorators import timeit
from quant_models.applications.feature_mining.feature_selection import train_features
from sklearn import decomposition

logger = Logger(log_level='DEBUG', handler='ch').get_log()
config = get_config()
feature_model_name = 'random_forest'


def get_selected_features(start_date=None, end_date=None, up_ratio=0.2, down_ratio=0.2):
    root = get_source_root()
    feature_source = os.path.join(os.path.realpath(root), 'data', 'features')
    files = os.listdir(feature_source)
    files = [item for item in files if item.startswith('score')]
    # get the score with the corresponding start and end date, otherwise return the latest one
    # TODO confirm whether the listdir function's return is sorted by time
    _path = 'score_{0}_{1}'.format(start_date, end_date) if 'score_{0}_{1}'.format(start_date, end_date) in files else \
        files[-1]
    _score_path = os.path.join(feature_source, _path)
    df = pd.read_csv(_score_path)
    df = df.sort_values(by='score')
    features = list(df['feature'])
    left, right = int(len(features) * up_ratio), int(len(features) * down_ratio)
    selected_features = features[:left]
    selected_features.extend(features[-right:])
    return list(set(selected_features))


@timeit
def train_models(model_name='', start_date='20140603', end_date='20181231', score_bound=(0.2, 0.1), bc='000300.XSHG'):
    '''

    :param model_name:
    :param start_date:
    :param end_date:
    :param score_bound: (up_ratio, down_ratio), will pick the features with socre in the top up_ratio ranking
    and in the bottom with down_ratio
    :return:
    '''
    m = Ml_Reg_Model(model_name)
    m = m.load_model(model_name) or m.build_model()
    root = get_source_root()

    df, score_df = train_features(start_date=start_date, end_date=end_date, bc=bc)
    score_df = score_df.sort_values(by='score', ascending=False)
    _feature_names = list(score_df['feature'])
    n_rows = len(_feature_names)
    feature_names = list(
        set(_feature_names[:int(n_rows * score_bound[0])]).union(set(_feature_names[int(n_rows * score_bound[1]):])))

    # select the features by the ic values
    # feature_names = get_selected_features(start_date=start_date, end_date=end_date, up_ratio=score_bound[0],
    #                                       down_ratio=score_bound[1])

    train_X = df[feature_names].values
    dataes = list(df['TRADE_DATE'])
    sec_ids = list(df['SECURITY_ID'])
    train_Y = df.iloc[:, -1]
    decom_ratio = float(config['defaults']['decom_ratio'])

    # PCA processing
    pca = decomposition.PCA(n_components=int(len(feature_names) * decom_ratio))
    train_X = pca.fit_transform(train_X)
    train_Y = train_Y.fillna(0.0)
    st = time.time()
    logger.info('start training the models')
    mse_scores, r2_scores = m.train_model(train_X[:1000], train_Y[:1000])
    et = time.time()
    logger.info('complete training model with time:{0}'.format(et - st))
    model_full_name = 'stock_selection_{0}'.format(model_name)
    m.save_model(model_full_name)
    logger.info("Mean squared error:{0}: %0.5f -  %0.5f" % (
        mse_scores.mean() - mse_scores.std() * 3, mse_scores.mean() + mse_scores.std() * 3))
    logger.info("Mean squared error:{0}: %0.5f -  %0.5f" % (
        r2_scores.mean() - r2_scores.std() * 3, r2_scores.mean() + r2_scores.std() * 3))
    result_path = os.path.join(os.path.join((os.path.join(os.path.realpath(root), 'data')), 'results'),
                               '{0}_{1}_{2}.txt'.format(model_name, start_date, end_date))
    with open(result_path, 'a+') as fout:
        fout.write('{0}\t{1}'.format(str(list(mse_scores)), str(list(r2_scores))))
    return mse_scores, r2_scores


if __name__ == '__main__':
    st = time.time()
    ret = train_models(model_name='linear', start_date='20150103', end_date='20190531', score_bound=(0.2, 0.1))
    et = time.time()
    print(et - st)
    print(ret)
