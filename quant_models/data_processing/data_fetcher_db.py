# -*- coding: utf-8 -*-
# @time      : 2019/1/3 12:11
# @author    : rpyxqi@gmail.com
# @file      : data_fetcher_db.py


import pyodbc
from collections import defaultdict

import pandas as pd

from quant_models.utils.logger import Logger
from quant_models.utils.oracle_helper import OracleHelper

logger = Logger(log_level='INFO', handler='ch').get_log()


class DataFetcherDB(object):
    def __init__(self):
        self.datayes_config = {"user": "gfangm", "pwd": "Gfangm1023_cms2019", "host": "10.200.40.170", "port": 1521,
                               "dbname": "clouddb",
                               "mincached": 0, "maxcached": 1}
        self._dyobj = OracleHelper(self.datayes_config)
        self._jyobj = pyodbc.connect('DRIVER={SQL Server};SERVER=172.21.6.196;DATABASE=JYDB;UID=yfeb;PWD=yfeb')

    def close(self):
        try:
            self._dyobj.close_conn()
        except Exception as ex:
            logger.error('Error for connection close with error:{0}'.format(ex))

    def get_industry_info(self):
        sql_str = 'SELECT ICSRS,TICKER_SYMBOL,EXCHANGE_CD FROM CUST.EQU_IPO'
        return self._dyobj.execute_query(sql_str)

    def get_idx_cons_dy(self, idx_id=None, index_date=None):
        '''

        :param idx_id: 1:上证综指； 1782：沪深300
        :return:
        '''
        sql_str = "SELECT B.TICKER_SYMBOL,B.EXCHANGE_CD FROM CUST.IDX_CONS A,CUST.MD_SECURITY B WHERE A.SECURITY_ID={0} " \
                  "AND A.INTO_DATE<=TO_DATE('{1}','YYYYMMDD') AND (A.OUT_DATE>TO_DATE('{1}','yyyymmdd') OR " \
                  "A.OUT_DATE is NULL) AND A.CONS_ID=B.SECURITY_ID".format(idx_id, index_date)
        logger.debug(sql_str)
        return self._dyobj.execute_query(sql_str)

    def get_idx_cons_jy(self, idx_security_id=''):
        '''

        :param idx_security_id:'000001':上证
        :param start_date:
        :param end_date:
        :return:
        '''
        sql_str = "SELECT A.SecuInnerCode,A.InDate,A.OutDate FROM JYDB.dbo.LC_IndexComponent A,JYDB.dbo.SecuMain B " \
                  "WHERE A.IndexInnerCode=B.InnerCode AND CONVERT(varchar(100), B.SecuCode, 112)='{}'".format(
            idx_security_id)
        logger.debug(sql_str)
        df = pd.read_sql(sql_str, con=self._jyobj)
        return list(df.values), list(df.columns)

    def get_sw_2nd(self, security_ids=()):
        sql_str = "SELECT * FROM CUST.SW_HANGYE_2ND "
        if security_ids:
            sql_str += 'WHERE'
            ticker_symbols = [item.split('.')[0] for item in security_ids]
            exchange_cds = [item.split('.')[1] for item in security_ids]
            if len(security_ids) > 1:
                sql_str = "{0} TICKER_SYMBOL IN {1} AND EXCHANGE_CD IN {2}".format(sql_str, tuple(ticker_symbols),
                                                                                   tuple(exchange_cds))
            else:
                sql_str = "{0} TICKER_SYMBOL = '{1}' AND EXCHANGE_CD = '{2}'".format(sql_str, ticker_symbols[0],
                                                                                     exchange_cds[0])
        return self._dyobj.execute_query(sql_str)

    def get_idx_weights(self, trade_date='', idx_id=None):
        '''
        :param trade_date:
        :param idx_id:  1:上证综指； 1782：沪深300
        :return:
        '''
        sql_str = "SELECT UNIQUE(EFF_DATE) FROM CUST.idx_weight WHERE SECURITY_ID={0} AND EFF_DATE<=TO_DATE({1},'yyyymmdd') " \
                  "ORDER BY EFF_DATE DESC".format(idx_id, trade_date)
        rows, desc = self._dyobj.execute_query(sql_str)
        eff_date = rows[0][0].strftime('%Y%m%d')
        sql_str = "SELECT A.TICKER_SYMBOL,A.EXCHANGE_CD,B.WEIGHT " \
                  "FROM CUST.MD_SECURITY A JOIN CUST.IDX_WEIGHT B ON A.SECURITY_ID = B.CONS_ID " \
                  "WHERE B.SECURITY_ID={0} AND B.EFF_DATE=TO_DATE({1},'yyyymmdd')".format(idx_id, eff_date)
        return self._dyobj.execute_query(sql_str)

    def get_adjust_dates(self, idx_id=None, start_date='', end_date=''):
        sql_str = "SELECT UNIQUE(INTO_DATE) FROM CUST.IDX_CONS WHERE SECURITY_ID={0} AND INTO_DATE>=" \
                  "TO_DATE('{1}','YYYYMMDD') AND INTO_DATE<=TO_DATE('{2}','YYYYMMDD')".format(idx_id, start_date,
                                                                                              end_date)
        return self._dyobj.execute_query(sql_str)

    def get_equ_factor(self, factor_type='', security_ids=(), fields=None, start_date=None, end_date=None):
        '''
        :param factor_type: str
        :param security_ids:list
        :param fields: list
        :param start_date: str, 'yyyymmdd'
        :param end_date: str, 'yyyymmdd'
        :return:
        '''
        table_name = 'cust.equ_factor_{0}'.format(factor_type)
        if fields:
            fields.extend(['SECURITY_ID', 'TRADE_DATE'])
            fields = list(set(fields))
            sql_str = 'SELECT ' + ','.join(fields) + ' FROM {0}'.format(table_name)
        else:
            sql_str = "SELECT * FROM {0}".format(table_name)
        if security_ids or start_date or end_date:
            sql_str += ' WHERE'
        if security_ids:
            if len(security_ids) > 1:
                sql_str = "{0} SECURITY_ID IN {1}".format(sql_str, tuple(security_ids))
            else:
                sql_str = "{0} SECURITY_ID = '{1}'".format(sql_str, security_ids[0])
        if start_date and end_date and start_date == end_date:
            sql_str = "{0} AND TRADE_DATE = {1}".format(sql_str, start_date)
        else:
            if start_date:
                sql_str = "{0} AND TRADE_DATE >= {1}".format(sql_str, start_date)
            if end_date:
                sql_str = "{0} AND TRADE_DATE < {1}".format(sql_str, end_date)
        logger.debug('Execute query: {0}'.format(sql_str))
        print(sql_str)
        return self._dyobj.execute_query(sql_str)

    def get_mkt_equd(self, security_ids=(), fields=None, start_date=None, end_date=None, asset_type='stock'):
        '''

        :param security_ids:('ticker_symbol.exchange_cd')
        :param fields:
        :param start_date:
        :param end_date:
        :param asset_type: 'stock'|'idx'
        :return:
        '''
        table_name = {
            'stock': 'CUST.MKT_EQUD',
            'idx': 'CUST.MKT_IDXD',
            'future': 'CUST.MKT_FUTD'
        }.get(asset_type.lower())
        if fields:
            fields.extend(['TICKER_SYMBOL', 'EXCHANGE_CD', 'TRADE_DATE'])
            fields = list(set(fields))
            sql_str = 'SELECT ' + ','.join(fields) + ' FROM {0}'.format(table_name)
        else:
            sql_str = "SELECT * FROM {0}".format(table_name)
        if security_ids or start_date or end_date:
            sql_str += ' WHERE'
        if security_ids:
            ticker_symbols = [item.split('.')[0] for item in security_ids]
            exchange_cds = [item.split('.')[1] for item in security_ids]
            if len(security_ids) > 1:
                sql_str = "{0} TICKER_SYMBOL IN {1} AND EXCHANGE_CD IN {2}".format(sql_str, tuple(ticker_symbols),
                                                                                   tuple(exchange_cds))
            else:
                sql_str = "{0} TICKER_SYMBOL = '{1}' AND EXCHANGE_CD = '{2}'".format(sql_str, ticker_symbols[0],
                                                                                     exchange_cds[0])
        if start_date:
            sql_str = "{0} AND TRADE_DATE >= TO_DATE({1}, 'yyyymmdd')".format(sql_str, start_date)
        if end_date:
            sql_str = "{0} AND TRADE_DATE < TO_DATE({1}, 'yyyymmdd')".format(sql_str, end_date)
        logger.debug('Execute query: {0}'.format(sql_str))
        print(sql_str)
        return self._dyobj.execute_query(sql_str)

    def get_mkt_mins(self, startdate='', enddate='', sec_codes=[], filter='',
                     orderby='', groupby='', table_name='CUST.EQUITY_PRICEMIN'):
        '''
        Fetch the minute level data from tonglian in oracle
        Return the rows of values: ['DATADATE', 'TICKER', 'EXCHANGECD', 'SHORTNM', 'SECOFFSET', 'BARTIME', 'CLOSEPRICE',
         'OPENPRICE', 'HIGHPRICE', 'LOWPRICE', 'VOLUME', 'VALUE', 'VWAP']
         e.g. [(20180605, 600237, 'XSHG', '铜峰电子', 11640, '11:14', 4.46, 4.47, 4.47, 4.46, 68600, 306094.0, 4.462000000000001)]
        :param startdate: int
        :param enddate: int
        :param sec_codes: tuple of str
        :param filter:
        :param orderby:
        :param groupby:
        :return: rows, col_names; rows: list of tuple from the results; col_names: list of strings of the colume name of
                the table
        '''
        if not self.db_obj:
            logger.error("Fail in get_market_mins for empty db_obj")
        rows, col_names = self.get_dates_statics(startdate, enddate)
        all_trading_dates = [item[1].strftime('%Y%m%d') for item in rows if item[3] == 1]
        grouped_dates = defaultdict(list)

        for d in all_trading_dates:
            yymm = d[:6]
            rows, columns = [], []
            grouped_dates[yymm].append(int(d))
        total_len = len(grouped_dates)
        cnt = 0
        logger.info("Start query data in get_market_mins for query_date:{0}".format(len(grouped_dates)))
        for k, v in grouped_dates.items():
            cnt += 1
            logger.debug("query the {0} th table {1} out of {2}".format(cnt, k, total_len))
            v = sorted(v)
            sqlstr = self._get_sql_query(v[0], v[-1], sec_codes, filter, orderby,
                                         groupby, table_name)
            tmp_rows, desc = self.db_obj.execute_query(sqlstr)
            columns = [item[0] for item in desc]
            rows.extend(tmp_rows)
        logger.info("Done query data in get_market_mins for query_date:{0}".format(len(grouped_dates)))
        return rows, columns

    def get_report_contents(self):
        pass

    def get_annoucement_profitability(self, stock_ids=(), start_date='', end_date=''):
        sql_str = 'SELECT * FROM CUST.ANNOUNCEMENT_PROFITABILITY '
        if start_date:
            sql_str = "{0} WHERE PUBLISH_DATE >= TO_DATE({1}, 'yyyymmdd')".format(sql_str, start_date)
        if end_date:
            sql_str = "{0} AND PUBLISH_DATE < TO_DATE({1}, 'yyyymmdd')".format(sql_str, end_date)
        if stock_ids:
            # ticker_symbols = [item.split('.')[0] for item in security_ids]
            # exchange_cds = [item.split('.')[1] for item in security_ids]
            if len(stock_ids) > 1:
                sql_str = "{0} AND  STOCKID IN {1}".format(sql_str, tuple(stock_ids))
            else:
                sql_str = "{0} AND STOCKID = '{1}'".format(sql_str, stock_ids[0])
        logger.debug('Execute query: {0}'.format(sql_str))
        return self._dyobj.execute_query(sql_str)

    def get_halt_info(self, security_ids=''):
        if isinstance(security_ids, (list, tuple)):
            if len(security_ids) > 1:
                ticker_symbols = [item.split('.')[0] for item in security_ids]
                exchange_cds = [item.split('.')[1] for item in security_ids]
                sql_str = "SELECT TICKER_SYMBOL,EXCHANGE_CD,HALT_BEGIN_TIME, RESUMP_BEGIN_TIME FROM cust.md_sec_halt " \
                          "WHERE TICKER_SYMBOL IN {0} AND EXCHANGE_CD IN {1}".format(
                    tuple(ticker_symbols), tuple(exchange_cds))
                return self._dyobj.execute_query(sql_str)

        security_ids = security_ids[0]
        ticker, exchangecd = security_ids.split('.')
        sql_str = "SELECT TICKER_SYMBOL,EXCHANGE_CD,HALT_BEGIN_TIME, RESUMP_BEGIN_TIME FROM cust.md_sec_halt " \
                  "WHERE TICKER_SYMBOL='{0}' AND EXCHANGE_CD='{1}'".format(
            ticker, exchangecd)
        logger.debug('Execute query: {0}'.format(sql_str))
        return self._dyobj.execute_query(sql_str)

    def get_data_by_sql(self, sql_str=""):
        return self._dyobj.execute_query(sql_str)

    def get_indust_stats(self):
        sql_str = "SELECT TO_CHAR(STATS_TYPE_CLASS) INDUST_NAME,TRADE_DATE,STATS_TYPE_CLASS_CD INDUST_CD, " \
                  "SUM(TURNOVER_VALUE) TURNOVER_VALUE, SUM(TURNOVER_VOL) TURNOVER_VOL " \
                  "FROM CUST.mkt_stats_ex_td_shsz WHERE STATS_TYPE_CD = 2 GROUP BY TRADE_DATE,STATS_TYPE_CLASS,STATS_TYPE_CLASS_CD" \
                  "ORDER BY TRADE_DATE"
        return self._dyobj.execute_query(sql_str)

    # FIXME double check the update industry information, check XGRQ field in table LC_ExgIndustry;
    # FIXME  check the updates. and fix some miss, e.g. 001979.XSHE and 603612.XSHG
    def get_sw_indust(self):
        sql_str = """
        SELECT B.SecuCode,B.ChiName,B.ChiNameAbbr,B.SecuMarket,A.FirstIndustryCode, A.FirstIndustryName
        FROM JYDB.dbo.LC_ExgIndustry A, JYDB.dbo.SecuMain B WHERE A.CompanyCode=B.CompanyCode AND A.Standard=24 AND
        A.IfPerformed=1 AND B.SecuMarket IN (83,90)
        """
        df = pd.read_sql(sql_str, con=self._jyobj)
        return list(df.values), list(df.columns)

    def get_security_codes(self, sec_type=1):
        """

        :param sec_type: 1 for A shares;
        :return:
        """
        sql_str = """
        SELECT * FROM JYDB.dbo.SecuMain WHERE SecuMarket IN (83,90) AND SecuCategory={} AND ListedState=1
        """.format(sec_type)
        df = pd.read_sql(sql_str, con=self._jyobj)
        return list(df.values), list(df.columns)

    def get_const(self, lb=1177):
        """
        :param lb:1177,证券主表；
        :return:
        """
        sql_str = """
        SELECT DM const_key,MS const_val FROM JYDB.dbo.CT_SystemConst WHERE LB={}
        """.format(lb)
        df = pd.read_sql(sql_str, con=self._jyobj)
        return list(df.values), list(df.columns)

    def _get_sec_main(self, inner_codes=[]):
        _ = '({0})'.format(','.join([str(item) for item in inner_codes]))
        sql_str = """SELECT SecuCode,SecuMarket FROM JYDB.dbo.SecuMain WHERE InnerCode IN {}""".format(_)
        df = pd.read_sql(sql_str, con=self._jyobj)
        return list(df.values), list(df.columns)

    def get_indust_mkt_jy(self, idx_codes=[], start_date=None, end_date=None, return_df=False):
        _ = '({0})'.format(','.join([str(item) for item in idx_codes]))
        sql_str = """SELECT * FROM JYDB.dbo.QT_IndexQuote WHERE InnerCode IN {}""".format(_)
        if start_date:
            sql_str = """{0} AND TradingDay >= '{1}'""".format(sql_str, start_date)
        if end_date:
            sql_str = """{0} AND TradingDay < '{1}'""".format(sql_str, end_date)
        logger.debug('Execute query: {0}'.format(sql_str))
        df = pd.read_sql(sql_str, con=self._jyobj)
        if return_df:
            return df
        return list(df.values), list(df.columns)

    def get_sw_idx_codes_jy(self):
        sql_str = """SELECT A.IndexCode,A.IndustryCode,B.ChiName FROM JYDB.dbo.LC_CorrIndexIndustry A, JYDB.dbo.SecuMain B WHERE 
        A.IndexCode=B.InnerCode AND A.IndustryStandard=9 AND A.IndustryCode
        IN (SELECT DISTINCT(FirstIndustryCode) FROM JYDB.dbo.CT_IndustryType WHERE Standard=9 AND Classification=1)
        AND CONVERT(varchar(100), B.SecuCode,112)<='801230'"""
        df = pd.read_sql(sql_str, con=self._jyobj)
        return list(df.values), list(df.columns)

    # FIXME only query the first code in security_ids
    def get_idx_mkt_jy(self, security_ids=[], start_date='', end_date=''):
        _ticker, _exchange_cd = security_ids[0].split('.')
        sql_str = """SELECT * FROM JYDB.dbo.QT_IndexQuote A, JYDB.dbo.SecuMain B WHERE SecuCategory=4 AND 
        CONVERT(varchar(100), SecuCode,112)='{0}' AND A.InnerCode=B.InnerCode AND A.TradingDay>='{1}' and 
        A.TradingDay<'{2}' AND B.SecuMarket IN (83,90)
        """.format(_ticker, start_date, end_date)
        print(sql_str)
        df = pd.read_sql(sql_str, con=self._jyobj)
        return list(df.values), list(df.columns)

    def get_equtiy_mkt_jy(self, security_ids=(), fields=None, start_date=None, end_date=None, asset_type='stock'):
        tickers = [item.split('.')[0] for item in security_ids]
        if tickers:
            _ = '('
            for t in tickers:
                _ = "{0}'{1}',".format(_,t)
            _ = _[:-1] + ')'
            sql_str = """SELECT * FROM JYDB.dbo.QT_DailyQuote A, JYDB.dbo.SecuMain B WHERE SecuCategory=1 AND
            CONVERT(varchar(100), SecuCode,112) IN {0} AND A.InnerCode=B.InnerCode AND A.TradingDay>='{1}' and A.TradingDay<'{2}'
            AND B.SecuMarket IN (83,90)
                """.format(_, start_date, end_date)
            print(sql_str)
            df = pd.read_sql(sql_str, con=self._jyobj)
            return list(df.values), list(df.columns)


