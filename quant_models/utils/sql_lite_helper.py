# -*- coding: utf-8 -*-
# @time      : 2018/12/25 19:57
# @author    : rpyxqi@gmail.com
# @file      : sql_lite_helper.py


import sqlite3 as sqlite
import os
from quant_models.utils.io_utils import load_json_file
import cx_Oracle
import traceback
from quant_models.utils.helper import get_source_root


class SQLiteHelper(object):
    def __init__(self):
        root = get_source_root()
        # get the file name of the features
        feature_source = os.path.join(os.path.realpath(root), 'data', 'features', 'cache_features')
        try:
            self._conn = sqlite.connect(feature_source)
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


# CREATE FEATURE TABLE
def create_features_table():
    db = SQLiteHelper()
    root = get_source_root()
    # get the file name of the features
    feature_mapping_source = os.path.join(os.path.realpath(root), 'conf', 'feature_mapping.json')
    feature_mapping = load_json_file(feature_mapping_source)
    _vals = list(feature_mapping.values())
    fields = []
    for item in _vals:
        fields.extend(item)
    table_name = 'FEATURE_CACHE'
    s1 = "CREATE TABLE {0} (TICKER_SYMBOL INT, TRADE_DATE TEXT,SECURITY_ID TEXT,D_LABEL REAL,M_LABEL REAL, ".format(
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
    #delete feature table
    # db = SQLiteHelper()
    # db.execute_sql("DROP TABLE FEATURE_CACHE")
    # db.execute_query("SELECT * FROM FEATURE_CACHE")

    #create feature table
    create_features_table()

    # _conn = cx_Oracle.connect('cust/admin123@10.200.40.170/clouddb', encoding='utf-8')
    # cursor = _conn.cursor()
    # sql_str = "SELECT * FROM equ_factor_vs WHERE TRADE_DATE>20181201"
    # cursor.execute(sql_str)
    # results = cursor.fetchall()
    # desc = cursor.description
    # cursor.close()
    # print(desc)
    # db = SQLiteHelper()
    # create_features_tables()
    # db.execute_sql("DROP TABLE FEATURE_CACHE")
    # db.execute_query("SELECT * FROM FEATURE_CACHE")
