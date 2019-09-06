# -*- coding: utf-8 -*-
# @time      : 2019/1/22 14:24
# @author    : rpyxqi@gmail.com
# @file      : hedge_rq.py

from rqalpha.api import *
from rqdatac import *
import rqdatac as rqd
from quant_models.utils.decorators import timeit
from collections import defaultdict
import pandas as pd


def log_cash(context, bar_dict):
    logger.info('Remaining cash:{0}'.format(context.portfolio.cash))


def get_history():
    history_bars('000001.XSHE', 4, '1d', 'close', skip_suspended=True, include_now=True)


def init(context):
    context.slippage = 0.5
    context.commission = 0.02

    scheduler.run_daily(log_cash)


@timeit
def query_fundamental(context):
    trade_date = context.now.strftime('%Y%m%d')
    # func_df = get_fundamentals(
    #     query(
    #         fundamentals.eod_derivative_indicator.pe_ratio
    #     ).filter(
    #         fundamentals.stockcode in context.stock_lst
    #     ).order_by(
    #         fundamentals.eod_derivative_indicator.pe_ratio
    #     ),
    #     trade_date
    # )
    func_df = get_fundamentals(
        query(
            fundamentals.eod_derivative_indicator.pe_ratio
        ).order_by(
            fundamentals.eod_derivative_indicator.pe_ratio
        ),
        trade_date
    )
    pe_df = func_df['pe_ratio']
    stocks = list(pe_df.columns)
    _ = pe_df.values
    # values = [item[0] for item in _]
    context.cached = {trade_date: list(zip(stocks, _[0]))}
    # update_universe(context.cached)


def _is_suspended(sec_id=None, trade_date=''):
    ret = is_suspended(sec_id, start_date=trade_date, end_date=trade_date).values
    return ret[0][0]


def before_trading(context, bar_dict):
    rqd.init('license',
             'BZPntqmgrLH4lKAOjx671eO91m87PaKGRLpXoldWGCf-3Skw8oZ_VnjVSA-M4fRtcuye-HfX8velChxb0iLxDtpzt9DeTCVJsdSYOKQ4'
             'grrpqge2IYBUMKmrpFqMiuI6ZXw5d9eje7Qtu28Z-wgDKeWpoI4K0MNYf2aPt_ejSsM=MABNzwICqZ_LX_sk549IAzaYHdtWgAJwSkyRAe'
             'DQUq992HgGBL1nC_WicztRvUqa-WiwR-_cGj8-vfFakdVXqtNx9tapTpSuM5PKhfTfDJqOD7HH3jr2KpgI_IX4gNDbKT0a4W9mCKz6qka7'
             'KDTj_TlCpdV5_lIfAd7UaIgm9pc=',
             ('115.159.129.195', 443), proxy_info=('http', '172.20.0.2', 8086, 'yuxiaoqi', '0814xqYu'))
    trade_date = context.now.strftime('%Y-%m-%d')
    # print(trade_date)
    context.stock_lst = index_components('000300.XSHG', trade_date)
    # print(context.stock_lst)
    # factor_ret = rqd.get_factor_exposure(id_or_symbols=context.stock_lst, start_date=trade_date,
    #                                      end_date=trade_date,
    #                                      factors=['momentum'])
    # indexes = list(factor_ret.index)
    # values = list(factor_ret['momentum'])
    #
    # context.cached = defaultdict(list)
    # for idx, val in enumerate(indexes):
    #     book_id, date = val
    #     factor = values[idx]
    #     context.cached[date.strftime('%Y%m%d')].append((book_id, factor))
    query_fundamental(context)
    context.b_price = get_price('000300.XSHG', start_date=trade_date, end_date=trade_date, fields='close').values[0]
    context.mtm = context.b_price * 300


def handle_bar(context, bar_dict):
    trade_date = context.now.strftime('%Y%m%d')
    factor_lst = context.cached.get(trade_date)
    factor_lst.sort(key=lambda x: x[1])
    cnt = 0
    order_stocks = []
    for item in factor_lst:
        if cnt >= 10 or item[0] not in context.stock_lst:
            break
        if not _is_suspended(item[0], trade_date):
            cnt += 1
            order_stocks.append(item[0])
    # print(len(order_stocks))
    for s in order_stocks:
        # print("order stock", s)
        # order_target_percent(s, 0.1)
        logger.info('Order value:{0} for stock:{1}'.format(context.mtm / 10, s))
        order_value(s, context.mtm / 10)
        # order_shares(s, 100)
        # print(context.portfolio.positions[s])
        # print(context.portfolio.cash)
    # print(context.portfolio.cash)
    if not context.portfolio.future_account.positions:
        sell_open('IF88', 1)
    # print('future account', context.portfolio.future_account.positions)
    # print(pd.read_pickle("result.pkl"))


def after_trading(context):
    pass
    # print(context.portfolio.cash)


__config__ = {

    "base": {
        "start_date": "2018-01-03",
        "end_date": "2018-12-31",
        "frequency": "1d",
        "matching_type": "current_bar",
        "data_bundle_path": "rqdata/bundle",
        "margin_multiplier": 0.10,
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
            "enabled": False,
            "show": True,
        },
        "sys_analyser": {
            "enabled": True,
            "show": True,
            "plot": True,
            "output_file": "output_result.pkl",
            "plot":True,
            "plot_save_file":'output_result.png',
        },
        "sys_simulation": {
            "enabled": True,
            "priority": 100,
            "slippage": 0.02,
            "commission_multiplier":0.0008,
        },
        "sys_risk":{
            "enabled":True,
            "validate_position":False,
        }
    },
}
