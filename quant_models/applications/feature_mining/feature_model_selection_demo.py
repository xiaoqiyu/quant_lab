# -*- coding: utf-8 -*-
# @time      : 2019/10/26 15:48
# @author    : rpyxqi@gmail.com
# @file      : feature_model_selection_demo.py

import time
from quant_models.applications.feature_mining.feature_selection import feature_selection_ic
from quant_models.applications.feature_mining.model_selection import train_stock_selection


def main():
    # calculate ic score and save features
    ret = feature_selection_ic(start_date='20190103', end_date='20190531', data_source=0,
                               feature_types=[], train_feature=True, saved_feature=True,
                               bc='000300.XSHG')

    # train ml model
    st = time.time()
    ret = train_stock_selection(model_name='linear', start_date='20150103', end_date='20190531', score_bound=(0.2, 0.1))
    et = time.time()
    print(et - st)


if __name__ == '__main__':
    main()

