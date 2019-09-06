# -*- coding: utf-8 -*-
# @time      : 2019/1/3 12:12
# @author    : rpyxqi@gmail.com
# @file      : data_fetcher_cache.py

import pandas as pd
import os
import gc


class DataFetcherCache(object):
    def __init__(self):
        par_dir = os.path.dirname(os.path.dirname(__file__))
        self._cache_path = "{0}/{1}".format(par_dir, 'data')
        self._cache_path = "E:\pycharm\quant_github\quant_models\data"

    def get_equ_factor(self, factor_type='', security_ids=(), fields=None, start_date=None, end_date=None):
        '''
        Support factors for stocks of HS300 from 20160103 to 20181225 now
        :param factor_type:
        :param security_ids:
        :param fields:
        :param start_date:
        :param end_date:
        :return:
        '''
        # factor_df = pd.read_csv("{0}\\{1}_20160103_20181225.csv".format(self._folder, factor_type))
        file_name = '{0}_{1}_{2}.csv'.format(factor_type, start_date, end_date)
        feature_path = "{0}\\features\{1}".format(self._cache_path, file_name)
        # feature_path = os.path.join(os.path.join(self._cache_path, 'features'), file_name)
        # factor_df = pd.read_csv(feature_path)
        factor_df = pd.read_csv(
            'E:\pycharm\quant_github\quant_models\data\\features\{0}_20160103_20181225.csv'.format(factor_type))
        desc = list(factor_df.columns)
        fields = fields or desc

        fields.extend(['TRADE_DATE', 'SECURITY_ID'])
        fields = list(set(fields))
        factor_df = factor_df[fields]
        factor_df = factor_df[factor_df.TRADE_DATE >= start_date]
        factor_df = factor_df[factor_df.TRADE_DATE < end_date]
        lst = factor_df.values
        desc = list(factor_df.columns)
        sec_id_idx = desc.index('SECURITY_ID')
        del factor_df
        rows = [item for item in lst if item[sec_id_idx] in security_ids]
        del lst
        gc.collect()
        return rows, desc

    def get_mkt_equd(self, security_ids=(), fields=None, start_date=None, end_date=None, asset_type='stock',
                     source='file'):
        if source == 'file':
            pass

    def get_idx_cons(self, idx_id=None, ticker=None, index_date=None):
        '''

        :param idx_id: 1:上证综指； 1782：沪深300
        :return:
        '''
        pass



