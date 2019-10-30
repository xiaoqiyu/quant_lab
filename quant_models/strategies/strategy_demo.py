# -*- coding: utf-8 -*-
# @time      : 2018/11/1 15:09
# @author    : rpyxqi@gmail.com
# @file      : strategy_demo.py

import pandas as pd
import pprint
import os
import time
from rqalpha import run_file
import matplotlib.pyplot as plt
import datetime as dt

# from quant_models.strategies.feature_mining_strategy import __config__
from quant_models.strategies.s_buy_and_hold import __config__
from quant_models.utils.logger import Logger
from quant_models.utils.decorators import timeit
from quant_models.utils.helper import get_source_root

logger = Logger('log.txt', 'INFO', __name__).get_log()


def result_analysis():
    result = pd.read_pickle('output_result.pkl')
    pprint.pprint(result)
    # margin_lst = list(result['future_positions']['margin'])
    for k, v in list(result.items())[1:]:
        r = get_source_root()
        result_path = os.path.join(r, 'data', 'results', '{0}.csv'.format(k))
        print('save the result to:{0}'.format(result_path))
        v.to_csv(result_path)
    # plt.plot(margin_lst)
    # plt.show()


@timeit
def run_strategy(strategy_file_path="hedge_rq.py"):
    run_file(strategy_file_path, __config__)


if __name__ == '__main__':
    run_strategy("s_buy_and_hold.py")
    # result_analysis()
