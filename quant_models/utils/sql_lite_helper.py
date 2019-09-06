# -*- coding: utf-8 -*-
# @time      : 2018/12/25 19:57
# @author    : rpyxqi@gmail.com
# @file      : sql_lite_helper.py


import sqlite3 as sqlite
import pandas as pd
import datetime
import traceback
from quant_models.utils.helper import list_files
from quant_models.utils.io_utils import load_json_file
import cx_Oracle


class SQLiteHelper(object):
    def __init__(self):
        try:
            self._conn = sqlite.connect('E:/pycharm/algo_trading/quant_models/quant_models/data/features/cache_data')
        except Exception as ex:
            traceback.print_exc()

    def close_conn(self):
        try:
            self._conn.close()
        except Exception as ex:
            traceback.print_exc()

    def execute_query(self, sql):
        # conn = self._pool.connection()
        cursor = self._conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        desc = cursor.description
        descs = [item[0] for item in desc]
        cursor.close()
        return results, descs

    def execute_sql(self, sql):
        # conn = self._pool.connection()
        cursor = self._conn.cursor()
        cursor.execute(sql)
        self._conn.commit()
        # cursor.close()
        # self._conn.close()


def create_factor_tables():
    db = SQLiteHelper()
    factor_mappings = load_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\feature_mapping.json')
    # table_names = ['EQU_FACTOR_{0}'.format(item) for item in list(factor_mappings.keys())]

    for f_type, fields in factor_mappings.items():
        table_name = 'EQU_FACTOR_{0}'.format(f_type.upper())
        s1 = "CREATE TABLE {0} (SECURITY_ID_INT INT, SECURITY_ID TEXT,TRADE_DATE INT,TICKER_SYMBOL TEXT, ".format(
            table_name)
        for f in fields:
            s1 += "{0} REAL,".format(f)
        s1 = s1[:-1] + ')'
        print(s1)
        try:
            db.execute_sql(s1)
        except Exception as ex:
            print(ex)



if __name__ == '__main__':
    # _conn = cx_Oracle.connect('cust/admin123@10.200.40.170/clouddb', encoding='utf-8')
    # cursor = _conn.cursor()
    # sql_str = "SELECT * FROM equ_factor_vs WHERE TRADE_DATE>20181201"
    # cursor.execute(sql_str)
    # results = cursor.fetchall()
    # desc = cursor.description
    # cursor.close()
    # print(desc)
    db = SQLiteHelper()
