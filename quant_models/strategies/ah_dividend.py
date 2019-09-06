# -*- coding: utf-8 -*-
# @time      : 2019/9/4 21:42
# @author    : rpyxqi@gmail.com
# @file      : ah_dividend.py


# -*- coding: utf-8 -*-
# @time      : 2019/8/16 12:21
# @author    : yuxiaoqi
# @file      : hk_hongli.py

import pandas as pd
import talib
import pyodbc
import traceback
import numpy as np
from collections import defaultdict


class AH_Selection(object):
    def __init__(self):
        try:
            self.conn = pyodbc.connect('DRIVER={SQL Server};SERVER=172.21.6.196;DATABASE=JYDB;UID=yfeb;PWD=yfeb')
        except:
            traceback.print_exc()
        else:
            pass

    def get_raw_sec_codes(self):
        df = pd.read_excel('D:\strategy\AH\data\\ah_dividend_1519.xlsx', sheetname="万得")
        _sec_code = set((df['code']))
        return ['0{0}'.format(item.split('.')[0]) for item in _sec_code if isinstance(item, str)]

    def get_sec_code(self, divident_ratio=(0.3, 0.6)):
        sec_ps = defaultdict(list)
        sec_in = dict()
        sec_name = dict()
        for y in range(2015, 2019):
            path = 'D:\strategy\AH\data\pershare{0}.xlsx'.format(y)
            df = pd.read_excel(path)
            df = df[["证券代码", "名称", "每股盈利(摊薄)", "每股派息", "所属wind行业"]]

            for item in df.values:
                earning = float(item[2])
                dividend = float(item[3])
                if earning != earning or dividend != dividend:
                    continue
                if earning > 0 and (dividend / earning < divident_ratio[0] or dividend / earning > divident_ratio[1]):
                    continue
                _tmp = [earning, dividend, dividend / earning]
                _ = sec_ps.get(item[0]) or list()
                _.append(_tmp)
                sec_ps.update({item[0]: _})
                sec_in.update({item[0]: item[-1]})
                sec_name.update({item[0]: item[1]})
        # pprint.pprint(sec_ps)
        scores = []
        # print(sec_ps['1157.HK'])
        for sec_code, val in sec_ps.items():
            div_ratio = [item[2] for item in val]
            earnings = [item[0] for item in val]
            divs = [item[1] for item in val]
            scores.append([sec_code, sec_name.get(sec_code), round(sum(div_ratio) / len(div_ratio), 2),
                           round(sum(earnings) / len(earnings), 2),
                           round(sum(divs) / len(divs), 2), sec_in.get(sec_code)])
        n_scores = len(scores)
        # scores = sorted(scores, key=lambda x:x[2], reverse=True)[int(n_scores*divident_ratio[0]): int(n_scores*divident_ratio[1])]
        scores = sorted(scores, key=lambda x: x[3], reverse=True)[:int(n_scores * 0.5)]
        df = pd.DataFrame(scores, columns=['sec_code', 'sec_name'  'div_ratio', 'earning', 'divident', 'indu'])
        df.to_csv('ah_data/score.csv')
        # print(n_scores, len(scores))
        # pprint.pprint(scores)
        sec_codes = ['0{0}'.format(item[0].split('.')[0]) for item in scores]
        return sec_codes

    def get_dividend_data(self, sec_codes=[]):
        _ = "("
        flag = True
        for item in sec_codes:
            if flag:
                flag = False
            else:
                _ = _ + ','
            _ = "{0}'{1}'".format(_, item)
        str = _ + ")"
        sqlstr = '''
            SELECT A.IfDividend,A.CashDividendPS,A.EndDate,B.SecuCode,ChiName,CONVERT(varchar(100),B.SecuAbbr,112) as
             SecName,CONVERT(varchar(100), B.SecuCode,112) as SecuCode FROM JYDB.dbo.HK_Dividend A, JYDB.dbo.HK_SecuMain B WHERE A.InnerCode=B.InnerCode AND EndDate>'2000-01-01' 
             AND CONVERT(varchar(100),B.SecuCode,112) in {}
        '''.format(str)
        df_div = pd.read_sql(sqlstr, con=self.conn)
        # print(df_div)
        df_div.fillna(0.0)

        sqlstr = '''
        SELECT  A.BeginDate,A.EndDate,A.FinancialYear,A.BasicEPS,A.DilutedEPS,CONVERT(varchar(100),B.SecuAbbr,112) as 
        SecName,CONVERT(varchar(100),B.SecuCode,112) as 
        SecuCode  FROM JYDB.dbo.HK_MainIndex A, JYDB.dbo.HK_SecuMain B WHERE A.CompanyCode=B.CompanyCode AND EndDate>'2000-01-01' 
        AND A.PeriodMark in (6,12)
        AND CONVERT(varchar(100),B.SecuCode,112) in {}
        '''.format(str)
        df_eps = pd.read_sql(sqlstr, con=self.conn)
        # print(df_eps)
        df_eps.fillna(0.0)

        rows = []
        _values = list(df_div.values)
        col_div = list(df_div.columns)
        col_eps = list(df_eps.columns)
        _val_div = [(item[1] if item[0] == 1 else 0.0, item[2], item[-2], item[-1]) for
                    item in df_div.values]
        _df_div = pd.DataFrame(_val_div, columns=['div', 'date', 'sec_name', 'sec_code'])
        _val_eps = [(item[1], item[3], item[-2], item[-1]) for item in df_eps.values]
        _df_eps = pd.DataFrame(_val_eps, columns=['date', 'eps', 'sec_name', 'sec_code'])
        dict_div = dict()
        dict_eps = dict()
        name_mapping = dict()
        # df_div [IfDividend,CashDividendPS,EndDate,SecuCode,ChiName,SecName,SecuCode]
        for item in _val_div:
            # _val_div: [cashdividend,enddate, secname, seccode]
            _date_key = '{0}{1}'.format(item[1].year, item[1].month) if item[1].month >= 10 else '{0}0{1}'.format(
                item[1].year, item[1].month)
            dict_div.update({'{0}_{1}'.format(item[-1], _date_key): item[0]})
            name_mapping.update({item[-1]: item[-2]})
        for item in _val_eps:
            # _val_eps:[enddate, eps, secname, seccode]
            _date_key = '{0}{1}'.format(item[0].year, item[0].month) if item[0].month >= 10 else '{0}0{1}'.format(
                item[0].year, item[0].month)
            dict_eps.update({'{0}_{1}'.format(item[-1], _date_key): item[1]})
            name_mapping.update({item[-1]: item[-2]})
        # pprint.pprint(dict_div)
        # pprint.pprint(dict_eps)
        dict_ratio = dict()
        sec_ratio = dict()
        for k, v in dict_eps.items():
            _v = dict_div.get(k) or 0.0
            try:
                r = round(float(_v / v), 2)
            except:
                continue
            dict_ratio.update({k: r})
            _sec_code, _d = k.split('_')
            _tmp = sec_ratio.get(_sec_code) or []
            _sec_name = name_mapping.get(_sec_code)
            _tmp.append([_v, v, r, _d])
            sec_ratio.update({_sec_code: _tmp})
            rows.append([_v, v, r, _d, _sec_code, _sec_name])
        df = pd.DataFrame(rows, columns=['dividend', 'eps', 'div_ratio', 'date', 'sec_code', 'sec_name'])
        df.to_csv('ah_data/jy_div.csv')

    def get_mkt_data(self, sec_codes=[], start_date='', end_date=''):
        _ = "("
        flag = True
        for item in sec_codes:
            if flag:
                flag = False
            else:
                _ = _ + ','
            _ = "{0}'{1}'".format(_, item)
        str = _ + ")"
        sqlstr = '''SELECT A.InnerCode, B.SecuCode,A.ClosePrice,A.TradingDay,CONVERT(varchar(100),B.SecuAbbr,112) as SecName FROM  JYDB.dbo.QT_HKDailyQuote A, JYDB.dbo.HK_SecuMain B
                   WHERE A.InnerCode=B.InnerCode AND CONVERT(varchar(100),B.SecuCode,112) in {0} AND A.TradingDay>='{1}' 
                   ORDER BY TradingDay'''.format(str, start_date)
        hk_price = pd.read_sql(sqlstr, con=self.conn)
        # print(sqlstr)
        hk_sec_names = list(set(hk_price["SecName"]))
        sec_names = [self._get_convert_name(val) for val in hk_sec_names]
        _ = "("
        flag = True
        for item in sec_names:
            if flag:
                flag = False
            else:
                _ = _ + ','
            _ = "{0}'{1}'".format(_, item)
        str = _ + ")"
        sqlstr1 = '''
          SELECT A.InnerCode, A.TradingDay,A.ClosePrice,CONVERT(varchar(10),B.ChiNameAbbr,112) as SecName,
          CONVERT(varchar(10),B.SecuCode,112 ) as SecCode FROM
          JYDB.dbo.QT_DailyQuote
          A, JYDB.dbo.SecuMain
          B
          WHERE
          A.InnerCode = B.InnerCode
          AND
          CONVERT(varchar(10), B.ChiNameAbbr, 112) in {0} AND A.TradingDay>='{1}' AND B.SecuCategory=1
                   ORDER BY TradingDay
          '''.format(str, start_date)
        print(sqlstr1)
        a_price = pd.read_sql(sqlstr1, con=self.conn)
        # print(a_price)
        hk_price.set_index('TradingDay')
        a_price.set_index('TradingDay')
        ret = hk_price.join(a_price, lsuffix='_hk', rsuffix='_a')
        # print(ret)
        ah_ratio = dict()
        for sec_name in sec_names:
            sec_hk = hk_price[hk_price.SecName == sec_name]
            sec_a = a_price[a_price.SecName == sec_name]
            _tmp = sec_a['ClosePrice'] / sec_hk['ClosePrice']
            ah_ratio.update({sec_name: _tmp})
        hk_price.to_csv('ah_data/hk.csv')
        a_price.to_csv('ah_data/a.csv')
        # pprint.pprint(ah_ratio)

    def div_analytics(self, ratio_bound=(0.3, 0.7), windows=10, features=['div_mastd', 'eps_mastd', 'ratio_mastd'],
                      top_k=30):
        df = pd.read_csv('ah_data/jy_div.csv', encoding='gbk')
        _sec_codes = list(df['sec_code'])
        dict_score = dict()
        sum_ma, sum_div, sum_ratio = 0.0, 0.0, 0.0
        cnt = 0
        name_mapping = dict(zip(df['sec_code'], df['sec_name']))
        for s in _sec_codes:
            _df = df[df.sec_code == s]
            _df = _df.sort_values(by='sec_code').fillna(0.0)
            n_rows = _df.shape[0]
            windows = windows if n_rows > windows else n_rows
            _div = _df['dividend']
            _eps = _df['eps']
            _div_ratio = _df['div_ratio']
            if list(_div_ratio)[-1] < ratio_bound[0] or list(_div_ratio)[-1] > ratio_bound[1]:
                continue
            ma_div = list(talib.MA(_div, windows))[-1]
            std_div = _div.std()
            ma_eps = list(talib.MA(_eps, windows))[-1]
            std_eps = _eps.std()
            ma_ratio = list(talib.MA(_div_ratio, windows))[-1]
            std_ratio = _div_ratio.std()
            # dict_score.update({s: [ma_div, ma_eps, ma_ratio, std_div, std_eps, std_ratio]})
            sum_ma += ma_div / std_div
            f_mappings = {'madiv': ma_div, 'maeps': ma_eps, 'ma_ratio': ma_ratio, 'div_mastd': ma_div / std_div,
                          'eps_mastd': ma_eps / std_eps, 'ratio_mastd': ma_ratio / std_ratio}
            _f_val = [f_mappings.get(item) for item in features]
            dict_score.update({s: _f_val})
        # pprint.pprint(dict_score)
        arr = np.array(list(dict_score.values()))
        _std = arr.std(axis=0)
        _m = arr.mean(axis=0)
        arr = (arr - _m) / _std
        final_score = list(arr.sum(axis=1))
        score_arr = list(zip(dict_score.keys(), final_score))
        score_arr = sorted(score_arr, key=lambda x: x[1], reverse=True)
        with open('ah_data/score.txt', 'w') as fin:
            fin.writelines(features)
            fin.write('\n')
            for row in score_arr:
                fin.write('{0},{1}\n'.format(name_mapping.get(row[0]), row[1]))
        _format_func = lambda item: "{0}{1}".format('0' * (5 - len(str(item))), item)
        for item in score_arr[:30]:
            print(name_mapping.get(item[0]), item[1])
        return [_format_func(item[0]) for item in score_arr[:30]], [item[1] for item in score_arr[:top_k]]

    def get_buy_signal(self, top_k=10):
        ah_ratio = dict()
        df_hk = pd.read_csv('ah_data/hk.csv', encoding='gbk', index_col=['TradingDay'])
        df_a = pd.read_csv('ah_data/a.csv', encoding='gbk', index_col=['TradingDay'])

        hk_sec = set(df_hk['SecName'])
        buy_lst = []
        dates = []
        for sec_name in hk_sec:
            df1 = df_hk[df_hk.SecName == self._get_convert_name(sec_name)]
            df2 = df_a[df_a.SecName == sec_name]
            _df = df1.join(df2, lsuffix='_hk', rsuffix='_a')
            dates = list(_df.index)
            if _df.shape[0]:
                _df['ah_ratio'] = _df['ClosePrice_a'] / _df['ClosePrice_hk']
                ah_ratio.update({sec_name: list(_df['ah_ratio'])})
                _df.dropna(axis=1, how='all', inplace=True)
                _df.fillna(method='bfill', inplace=True)
                try:
                    # if _df['ah_ratio'].mean() < _df['ah_ratio'][-10:].mean() and _df['ah_ratio'][-10:].mean() > 1.28:
                    if list(talib.MA(_df['ah_ratio'], 120))[-1] < list(talib.MA(_df['ah_ratio'], 10))[-1]:
                        buy_lst.append(sec_name)
                except:
                    pass
        df = pd.DataFrame(list(ah_ratio.values()), index=list(ah_ratio.keys()))
        df = df.transpose()
        # df.set_index(dates)
        df['dates'] = pd.Series(dates)
        df.to_csv('ah_data/ah_ratio.csv')
        buy_lst = sorted(buy_lst, key=lambda x: x[1], reverse=True)[:top_k]
        with open('ah_data/buy_lst.txt', 'a') as fin:
            fin.write((',').join(buy_lst))
            fin.write("\n")
        print(buy_lst)
        # print(df)

    def _get_convert_name(self, val):
        _name = val.split("股份")[0]
        _mapping = {"丽珠医药": "丽珠集团", "长飞光纤光缆": "长飞光纤", "万科企业": "万科A", "HTSC": "华泰证券", "华能国际电力股份": "华能国际",
                    "南京熊猫电子股份": "南京熊猫", "福莱特玻璃": "福莱特"}
        return _mapping.get(_name) or _name


def main(*args, **kwargs):
    ratio_bound = kwargs.get('ratio_bound') or (0.05, 0.85)
    windows = kwargs.get('windows') or 10
    features = kwargs.get('features') or ['div_mastd', 'eps_mastd', 'ratio_mastd']
    start_date = kwargs.get('start_date') or '2019-01-01'
    score_top_k = kwargs.get('score_top_k') or 30
    result_top_k = kwargs.get('result_top_k') or 10
    obj = AH_Selection()
    ret = obj.get_raw_sec_codes()
    # query dividend and earning data
    obj.get_dividend_data(ret)
    # calculate the scores by featuers, and select to score_top_k stocks as candidate
    sec_codes, scores = obj.div_analytics(ratio_bound=ratio_bound, windows=windows,
                                          features=features, top_k=score_top_k)
    # query market data and calculate the ah premium
    obj.get_mkt_data(sec_codes=sec_codes, start_date=start_date)
    # select result_top_k final buy list from the score_top_k by long/short term ah, which is undervalue in hk market
    obj.get_buy_signal(top_k=result_top_k)


if __name__ == '__main__':
    main(features=['maratio'])
