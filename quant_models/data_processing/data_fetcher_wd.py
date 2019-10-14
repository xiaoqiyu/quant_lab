# -*- coding: utf-8 -*-
# @time      : 2019/1/25 16:22
# @author    : rpyxqi@gmail.com
# @file      : data_fetcher_wd.py

import os
import numpy as np
import pandas as pd
from WindPy import w
from quant_models.utils.helper import get_source_root

w.start()


def get_econ_data(start_date='', end_date='', fields=None):
    path = os.path.join(get_source_root(), 'data', 'macro_indicators.xlsx')
    df = pd.read_excel(path)
    ids = fields or list(df['指标ID'])
    print(len(ids))
    # ret = w.edb(','.join(ids), start_date, end_date, "Fill=Previous")
    ret = w.edb(codes="", beginTime=start_date, endTime=end_date, options="Fill=Previous")
    dates = [item.strftime('%Y%m%d') for item in ret.Times]
    return dates, ret.Data, ids
    # values = np.array(ret.Data).transpose()
    # return pd.DataFrame(values,index=dates).sort_index()


def market_visual(start_date='20180101', end_date='20190325'):
    ret = w.wsd("399005.SZ", ['Close'], beginTime=start_date, endTime=end_date)
    mm = list(ret.Data[0])
    ret = w.wsd("000300.SH", ['Close'], beginTime=start_date, endTime=end_date)
    i300 = list(ret.Data[0])
    ret = w.wsd("000050.SH", ['Close'], beginTime=start_date, endTime=end_date)
    i50 = list(ret.Data[0])
    ret = w.wsd("000001.SH", ['Close'], beginTime=start_date, endTime=end_date)
    i01 = list(ret.Data[0])
    dates = list(ret.Times)
    import matplotlib.pyplot as plt
    max_val = max(mm) if max(mm) > max(i300) else max(i300)
    plt.grid(c='b')
    plt.plot(dates, mm, label=u'min', linestyle='-')
    plt.plot(dates, i300, label=u'300', linestyle='-')
    plt.plot(dates, i50, label=u'50', linestyle='-')
    plt.plot(dates, i01, label=u'01', linestyle='-')
    # plt.gcf().autofmt_xdate()

    plt.legend(['zhongxiao', '300', '50', 'shangzheng'])
    plt.show()


def get_all_month_start_end_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyydddd'
    '''
    # FIXME check the missing trade cal table
    # rows, cols = get_dates_statics(start_date, end_date)
    # return list(set([(item[6].strftime('%Y%m%d'), item[7].strftime('%Y%m%d')) for item in rows]))
    from WindPy import w
    w.start()
    _start_date = start_date[:6] + '01'
    _start_date = w.tdaysoffset(-1, _start_date).Data[0][0]
    data = w.tdays(_start_date, end_date, "Period=M").Data[0]
    ret = []
    for item in data:
        _next_start = w.tdaysoffset(1, item).Data[0][0]
        ret.append(item.strftime('%Y%m%d'))
        ret.append(_next_start.strftime('%Y%m%d'))
    return ret[1:-1]


if __name__ == '__main__':
    # df = get_econ_data(start_date='20180101', end_date='20190101')
    # print(df)
    # market_visual(start_date='20150101', end_date='20190325')
    # start_date='20190103'
    # end_date='20190915'
    # _start_date=start_date[:6]+'01'
    # _start_date = w.tdaysoffset(-1, _start_date).Data[0][0]
    #
    # ret = w.tdays(_start_date, end_date, "Period=M")
    # # ret = w.tdaysoffset(1, '20190131')
    # print(ret)
    # print(ret.Data[0])
    # ret = get_all_month_start_end

    ret = w.wsd("881008.WI", "close,pct_chg", "20190920", "20190925", "")
    print(ret)
