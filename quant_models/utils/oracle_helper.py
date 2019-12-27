# -*- coding: utf-8 -*-
# @time    : 2018/9/10 17:15
# @author  : huangyu10@cmschina.com.cn
# @file    : oracle_helper.py

import cx_Oracle
from DBUtils.PooledDB import PooledDB


class OracleHelper(object):
    def __init__(self, params):
        self._pool = PooledDB(cx_Oracle,
                              user=params.get("user"),
                              password=params.get("pwd"),
                              dsn="%s:%s/%s" % (params.get("host"), params.get("port"), params.get("dbname")),
                              mincached=int(params.get("mincached")),
                              maxcached=int(params.get("maxcached")),
                              blocking=True,
                              threaded=True,
                              encoding='UTF-8')

    def execute_query(self, sql):
        print(sql)
        conn = self._pool.connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        _desc = cursor.description
        cursor.close()
        conn.close()
        desc = [item[0] for item in _desc]
        return results, desc

    def execute_sql(self, sql):
        conn = self._pool.connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()


if __name__ == '__main__':
    db_config = {"user": "gfangm", "pwd": "Gfangm1023_cms2019", "host": "10.200.40.170", "dbname": "clouddb",
                 "mincached": 0, "maxcached": 2, "port": 1521}
    jy_config = {"user": "yfeb", "pwd": "yfeb", "host": "172.21.6.196", "dbname": "JYDB",
                 "mincached": 0, "maxcached": 2, "port": 1433}
    obj = OracleHelper(db_config)

    # sqlstr = "SELECT * FROM REPORT_CONTENT WHERE create_dt>TO_DATE('20181001','yyyymmdd') AND create_dt<TO_DATE('20181008','yyyymmdd') "
    # sqlstr="SELECT trade_date,close_price FROM CUST.MKT_EQUD WHERE ticker_symbol=603612 and trade_date>TO_DATE('20180718', 'yyyymmdd') and  trade_date<TO_DATE('20180720', 'yyyymmdd')"
    # sqlstr = "select * from cust.md_trade_cal where EXCHANGE_CD in ('XSHE','XSHG') and CALENDAR_DATE>= TO_DATE(20190701, 'YYYYMMDD') and CALENDAR_DATE<= TO_DATE(20190701,'YYYYMMDD')"
    # ret = obj.execute_query(sqlstr)
    # # print(ret)
    # rows = []
    # for item in ret[0]:
    #     try:
    #         if item[5].strftime('%Y%m%d') != item[6].strftime('%Y%m%d'):
    #             rows.append(item)
    #     except Exception as ex:
    #         pass
    # pprint.pprint(rows)
    # pprint.pprint(ret[0][0][2].read())
    ret = obj.execute_query("SELECT * FROM CUST.MKT_EQUD WHERE ROWNUM<10")
    # ret = obj.execute_query("SELECT TOP 10 * FROM JYDB.dbo.SecuMain")
    print(len(ret))
