# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : rpyxqi@gmail.com
# @file      : date_utils.py

import datetime
from quant_models.utils.logger import Logger
from quant_models.utils.oracle_helper import OracleHelper
from quant_models.data_processing.data_fetcher import DataFetcherDB
from quant_models.utils.helper import get_config

logger = Logger('log.txt', 'INFO', __name__).get_log()
df = DataFetcherDB()
config = get_config()


def get_dates_statics(start_date='', end_date='', calendar_date=''):
    db_obj = OracleHelper(config['datayes_db_config'])
    if calendar_date:
        sql_str = ('''select *
                  from cust.md_trade_cal
                  where EXCHANGE_CD in ('XSHE','XSHG') and CALENDAR_DATE=TO_DATE({}, 'YYYYMMDD')''').format(
            calendar_date)
    else:
        sql_str = ('''select *
                          from cust.md_trade_cal
                          where EXCHANGE_CD in ('XSHE','XSHG') and CALENDAR_DATE>= TO_DATE({}, 'YYYYMMDD') and CALENDAR_DATE<= TO_DATE({},'YYYYMMDD')''').format(
            start_date, end_date)
    ret, desc = db_obj.execute_query(sql_str)
    # cols = [item[0] for item in desc]
    return ret, desc


def datetime_delta(dt=None, format=None, days=0, hours=0, minutes=0, seconds=0, output_format=None):
    if isinstance(dt, datetime.datetime):
        return dt + datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    if isinstance(dt, str):
        if not format:
            logger.error('Format missing in datetime_delta for datetime:{0}'.format(dt))
            return None
        dt_time = datetime.datetime.strptime(dt, format)
        dt_time = dt_time + datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return dt_time.strftime(output_format) if output_format else dt_time.strftime(format)


def get_all_trading_dates(start_date='', end_date='', output_format='%Y-%m-%d'):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyymmdd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    output_format = output_format or '%Y-%m-%d'
    return sorted(list(set([item[1].strftime(output_format) for item in rows if item[3] == 1])))


def get_prev_trading_date(curr_date=''):
    rows, cols = get_dates_statics(curr_date, curr_date)
    return list(set([item[12].strftime('%Y%m%d') for item in rows]))


def get_all_month_end_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyydddd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    return list(set([item[7].strftime('%Y%m%d') for item in rows]))


def get_all_quarter_end_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyydddd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    return list(set([item[9].strftime('%Y%m%d') for item in rows]))


def get_all_year_end_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyymmdd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    return list(set([item[11].strftime('%Y%m%d') for item in rows]))


# TODO double check the start and end date pickup
def get_month_start_end_date(calendar_date=''):
    sql_str = "SELECT  MAX(MONTH_START_DATE),MIN(MONTH_END_DATE) FROM MD_TRADE_CAL WHERE calendar_date=TO_DATE('{0}', 'yyyymmdd')".format(
        calendar_date)
    rows, desc = df.get_data_by_sql(sql_str)
    return rows[0]


def get_week_start_end_date(calendar_date=''):
    sql_str = "SELECT  MAX(WEEK_START_DATE),MIN(WEEK_END_DATE) FROM MD_TRADE_CAL WHERE calendar_date=TO_DATE('{0}', 'yyyymmdd')".format(
        calendar_date)
    rows, desc = df.get_data_by_sql(sql_str)
    return rows[0]


def get_quarter_start_end_date(calendar_date=''):
    sql_str = "SELECT  MAX(QUARTER_START_DATE),MIN(QUARTER_END_DATE) FROM MD_TRADE_CAL WHERE calendar_date=TO_DATE('{0}', 'yyyymmdd')".format(
        calendar_date)
    rows, desc = df.get_data_by_sql(sql_str)
    return rows[0]


def get_year_start_end_date(calendar_date=''):
    sql_str = "SELECT  MAX(YEAR_START_DATE),MIN(YEAR_END_DATE) FROM MD_TRADE_CAL WHERE calendar_date=TO_DATE('{0}', 'yyyymmdd')".format(
        calendar_date)
    rows, desc = df.get_data_by_sql(sql_str)
    return rows[0]


def get_all_month_start_end_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyydddd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    return list(set([(item[6].strftime('%Y%m%d'), item[7].strftime('%Y%m%d')) for item in rows]))


# def is_trading_date(curr_date=None):
#     if curr_date:
#         start_date = datetime_delta(dt=curr_date, format='%Y-%m-%d', days=-31, output_format='%Y-%m-%d')
#         end_date = datetime_delta(dt=curr_date, format='%Y-%m-%d', days=31, output_format='%Y-%m-%d')
#         dates = get_all_trading_dates(start_date, end_date)
#         return curr_date in dates

if __name__ == '__main__':
    # ret = get_all
    # start_date, end_date = get_year_start_end_date('20181223')
    # print(start_date, end_date)
    ret = get_all_trading_dates('20190220', '20190620')
    import pandas as pd

    df = pd.DataFrame(ret, columns=['date'], index=None)
    # df.to_csv('trading_dates.csv')
    print(df)
