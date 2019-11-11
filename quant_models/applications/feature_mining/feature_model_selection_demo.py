# -*- coding: utf-8 -*-
# @time      : 2019/10/26 15:48
# @author    : rpyxqi@gmail.com
# @file      : feature_model_selection_demo.py

import time
import matplotlib.pyplot as plt
from quant_models.applications.feature_mining.feature_selection import cache_features
from quant_models.applications.feature_mining.feature_selection import train_features
from quant_models.applications.feature_mining.model_selection import train_models


def main():
    # cache_features run once only for setup
    # cache_features(start_date='20190103', end_date='20190531', data_source=0,
    #                feature_types=[], bc='000300.XSHG')

    # calculate the features scores
    df, score_df = train_features(start_date='20190103', end_date='20190531', bc='000300.XSHG')
    plt.plot(score_df['score'])
    plt.xlabel(score_df['feature'])
    plt.show()

    # train ml model
    # train_models(model_name='linear', start_date='20150103', end_date='20190531',
    #              score_bound=(0.2, 0.1))


if __name__ == '__main__':
    main()
