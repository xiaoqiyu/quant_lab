# -*- coding: utf-8 -*-
# @time      : 2019/5/22 15:12
# @author    : rpyxqi@gmail.com
# @file      : stock_selection_ml.py

import time
import os
import pandas as pd
from quant_models.model_processing.ml_reg_models import Ml_Reg_Model
from quant_models.utils.logger import Logger
from quant_models.utils.helper import get_config
from quant_models.utils.helper import get_source_root
from sklearn import decomposition

logger = Logger(log_level='DEBUG', handler='ch').get_log()
config = get_config()
feature_model_name = 'random_forest'


def train_stock_selection(model_name='', start_date='20140603', end_date='20181231'):
    m = Ml_Reg_Model(model_name)
    m.build_model()
    root = get_source_root()
    feature_path = os.path.join(os.path.realpath(root), 'conf', 'features_{0}_{1}.pkl'.format('20150103', '20151231'))
    df = pd.read_pickle(feature_path)
    feature_path = os.path.join(os.path.realpath(root), 'conf', 'features_{0}_{1}.pkl'.format('20160103', '20161231'))
    df = df.append(pd.read_pickle(feature_path))
    feature_path = os.path.join(os.path.realpath(root), 'conf', 'features_{0}_{1}.pkl'.format('20170103', '20171231'))
    df = df.append(pd.read_pickle(feature_path))
    feature_path = os.path.join(os.path.realpath(root), 'conf', 'features_{0}_{1}.pkl'.format('20180103', '20181231'))
    df = df.append(pd.read_pickle(feature_path))
    feature_path = os.path.join(os.path.realpath(root), 'conf', 'features_{0}_{1}.pkl'.format('20190103', '20190531'))
    df = df.append(pd.read_pickle(feature_path))
    col_names = list(df.columns)
    col_names.remove('TRADE_DATE')
    col_names.remove('SECURITY_ID')
    col_names.remove('LABEL')
    train_X = df[col_names].values
    dataes = list(df['TRADE_DATE'])
    sec_ids = list(df['SECURITY_ID'])
    train_Y = df.iloc[:, -1]
    decom_ratio = float(config['defaults']['decom_ratio'])
    n_cols = df.shape[1]
    pca = decomposition.PCA(n_components=int(n_cols * decom_ratio))
    train_X = pca.fit_transform(train_X)
    train_Y = train_Y.fillna(0.0)
    st = time.time()
    print('start training the models')
    mse_scores, r2_scores = m.train_model(train_X[:1000], train_Y[:1000])
    et = time.time()
    print('complete training model with time:{0}'.format(et - st))
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

    # ret_scores = defaultdict(dict)
    ret_scores = []
    for idx, text_x in enumerate(train_X):
        date = dataes[idx]
        sec_id = sec_ids[idx]
        # text_x = pca.fit_transform(list(item[:-3]))
        pred_y = m.predict(text_x)
        ret_scores.append([date, sec_id, pred_y[0]])
        # _d_dict = ret_scores.get(date) or {}
        # _d_dict.update({sec_id: float(pred_y[0])})
        # ret_scores.update({date: _d_dict})
    result_path = os.path.join(os.path.join((os.path.join(os.path.realpath(root), 'data')), 'results'),
                               'pred_score_{0}_{1}_{2}.csv'.format(model_name, start_date, end_date))
    _df = pd.DataFrame(ret_scores, columns=['date', 'sec_id', 'score'])
    _df.to_csv(result_path, index=False)
    # write_json_file(result_path, ret_scores)
    return mse_scores, r2_scores


if __name__ == '__main__':

    st = time.time()
    ret = train_stock_selection(model_name='poly_svr', start_date='20150103', end_date='20190531')
    et = time.time()
    print(et - st)
