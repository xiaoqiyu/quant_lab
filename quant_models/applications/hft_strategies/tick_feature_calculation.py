# -*- coding: utf-8 -*-
# @time      : 2019/12/9 19:33
# @author    : rpyxqi@gmail.com
# @file      : tick_feature_calculation.py


import uqer
from uqer import DataAPI

uqer_client = uqer.Client(token="26356c6121e2766186977ec49253bf1ec4550ee901c983d9a9bff32f59e6a6fb")


def get_tick_source_data():
    df = DataAPI.MktTicksHistOneDayGet(securityID=u"300634.XSHE", date='20191122', startSecOffset="", endSecOffset="",
                                       field=u"bidVolume1", pandas="1")
    print(df.columns)
    print(df['bidVolume1'].mean())


if __name__ == '__main__':
    get_tick_source_data()
