# -*- coding: utf-8 -*-
# @time      : 2018/12/20 16:34
# @author    : rpyxqi@gmail.com
# @file      : alpha_hedge.py

from quant_models.data_processing.data_fetcher import DataFetcher
import pprint

from rqalpha.api import *

df = DataFetcher()


def get_init_stocks(trade_date='', select_num=10):
    rows, desc = df.get_data_by_sql("SELECT TICKER_SYMBOL,EXCHANGE_CD FROM MKT_EQUD "
                                    "WHERE TRADE_DATE=TO_DATE({0},'yyyymmdd') AND PE IS NOT NULL "
                                    "AND ROWNUM<={1}  AND TICKER_SYMBOL IN ( SELECT MD_SECURITY.TICKER_SYMBOL"
                                    " FROM MD_SECURITY,IDX_CONS WHERE MD_SECURITY.SECURITY_ID = IDX_CONS.CONS_ID "
                                    "AND IDX_CONS.SECURITY_ID=1) ORDER BY PE DESC ".format(trade_date, select_num))
    return ['{0}.{1}'.format(item[0], item[1]) for item in rows]


def get_dominant_future_contracts():
    rows, desc = df.get_data_by_sql("SELECT TICKER_SYMBOL, EXCHANGE_CD,TRADE_DATE,CLOSE_PRICE FROM MKT_FUTD WHERE "
                                    "TURNOVER_VOL=some(SELECT MAX(turnover_vol) FROM MKT_FUTD WHERE TICKER_SYMBOL LIKE"
                                    " '%IF%' GROUP BY TRADE_DATE) ORDER BY TRADE_DATE ")
    pprint.pprint(rows)


def init(context):
    context.future = "IF88"
    context.stock = "000001.XSHE"
    subscribe(context.future)
    trade_date = context.now.strftime('%Y%m%d')
    context.stocks = get_init_stocks(trade_date=trade_date, select_num=10)
    # logger.info("Interested in: " + str(context.s1))


def handle_bar(context, bar_dict):
    # buy_open(context.future, 1)
    # sell_open(context.future, 1)
    each_order_percent = 0.8 / len(context.stocks)
    for sec_id in context.stocks:
        try:
            order_target_percent(sec_id, each_order_percent)
        except Exception as ex:
            print(ex, context.now, sec_id)


__config__ = {

    "base": {
        "start_date": "2016-01-09",
        "end_date": "2018-12-31",
        "frequency": "1d",
        "matching_type": "current_bar",
        "data_bundle_path": "rqdata/bundle",
        "benchmark": "000300.XSHG",
        "accounts": {
            "future": 1000000,
            "stock": 1000000
        }
    },
    "extra": {
        "log_level": "error",
    },

    "mod": {
        "sys_progress": {
            "enabled": True,
            "show": True,
        },
    },
}

if __name__ == '__main__':
    # get_dominant_future_contracts()
    ret = get_init_stocks(trade_date='20181220', select_num=50)
