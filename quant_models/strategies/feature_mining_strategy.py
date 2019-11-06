# -*- coding: utf-8 -*-
# @time      : 2019/10/28 20:41
# @author    : rpyxqi@gmail.com
# @file      : feature_mining_strategy.py


from rqalpha.api import *
import time
import os
import pandas as pd
from quant_models.utils.decorators import timeit
from quant_models.utils.helper import get_source_root
from quant_models.utils.helper import get_config
from quant_models.utils.helper import get_parent_dir
from quant_models.applications.feature_mining.model_selection import get_selected_features
from quant_models.applications.feature_mining.model_selection import train_models
from quant_models.applications.feature_mining.feature_selection import load_cache_features
from sklearn import decomposition
from sklearn.externals import joblib


model_name = 'linear'
config = get_config()
strategy_config = config['feature_mining_strategy']
# TODO change the path of the backtesting results
root = get_source_root()
# get the file name of the features
# _feature_path = os.path.join(os.path.realpath(root), 'data', 'features', 'feature_mining_strategy')
# w.start()


def init(context):
    model_path = os.path.join(get_parent_dir(), 'data', 'models', 'stock_selection_{0}'.format(model_name))
    feature_names = get_selected_features(__config__['base']['start_date'], __config__['base']['end_date'],
                                          up_ratio=0.2, down_ratio=0.1)
    print(feature_names)
    print('Start lodding features...')
    context.features = load_cache_features(__config__['base']['start_date'], __config__['base']['end_date'],
                                           __config__['base']['benchmark'])
    print('Complete lodding features...')
    context.model = joblib.load(model_path)
    context.feature_names = feature_names


def before_trading(context):
    pass


def handle_bar(context, bar_dict):
    now = context.now.strftime(config['constants']['standard_date_format'])
    feature_df = context.features[context.features.TRADE_DATE == int(now)]
    # #FIXME HACK FOR TESTING
    # feature_df = context.features[context.features.TRADE_DATE == list(context.features.TRADE_DATE)[0]]
    sec_ids = list(feature_df['SECURITY_ID'])
    selected_df = feature_df[context.feature_names]
    n_cols = selected_df.shape[1]
    decom_ratio = float(config['defaults']['decom_ratio'])
    pca = decomposition.PCA(n_components=int(n_cols * decom_ratio))
    train_X = pca.fit_transform(list(selected_df.values))
    pred_Y = context.model.predict(train_X)
    sec_scores = sorted(list(zip(sec_ids, pred_Y)), key=lambda x: x[1], reverse=True)
    buy_lst = [item[0] for item in sec_scores[:10]]
    for sec_id in buy_lst:
        order_target_percent(sec_id, 0.1)


def after_trading(context):
    pass
    # now = context.now.strftime(config['constants']['standard_date_format'])
    # next_tday = w.tdaysoffset(1, now).Data[0][0].strftime(config['constants']['standard_date_format'])
    # # now is the month end
    # if now[:6] != next_tday[:6]:
    #     start_date = w.tdaysoffset(-strategy_config['backtrack_period'], now).Data[0][0].strftime(
    #         config['constants']['standard_date_format'])
    #     end_date = w.tdaysoffset(-1, now).Data[0][0].strftime(config['constants']['standard_date_format'])
    #     train_models(model_name=strategy_config['model_name'], start_date=start_date, end_date=end_date, score_bound=(
    #         float(strategy_config['up_ratio']), float(strategy_config['down_ratio'])))


__config__ = {
    "base": {
        "start_date": strategy_config["start_date"],
        "end_date": strategy_config["end_date"],
        "frequency": "1d",
        "matching_type": "current_bar",
        "data_bundle_path": "rqdata/bundle",
        "benchmark": "000300.XSHG",
        "commission-multiplier": 1,
        "margin_multiplier": 1,
        "accounts": {
            "stock": 10000
        }
    },
    "extra": {
        "log_level": "error",
        "show": True,
    },
    "mod": {
        "sys_progress": {
            "enabled": False,
            "show": True,
        },
        "sys_analyser": {
            "enabled": True,
            "show": True,
            "plot": False,
            "output_file": "feature_mining_strategy.pkl",
            "plot": True,
            "plot_save_file": 'feature_mining_strategy.png',
        },
        "sys_simulation": {
            "enabled": True,
            "priority": 100,
            "slippage": 0.02,
            "commission_multiplier": 0.0008,
        },
        "sys_risk": {
            "enabled": True,
            "validate_position": False,
        }
    }
}
