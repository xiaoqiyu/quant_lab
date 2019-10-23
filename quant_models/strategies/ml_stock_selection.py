# -*- coding: utf-8 -*-
# @time      : 2018/11/1 17:09
# @author    : rpyxqi@gmail.com
# @file      : ml_stock_selection.py

from rqalpha.api import *
import numpy as np
import operator
import os
from collections import defaultdict
# import tushare as ts
from quant_models.model_processing.ml_reg_models import Ml_Reg_Model
from rqalpha.core.strategy_context import StrategyContext
from rqalpha.model.portfolio import Portfolio
import pandas as pd
from quant_models.utils.logger import Logger
from quant_models.utils.io_utils import load_json_file
from quant_models.utils.date_utils import datetime_delta
from quant_models.utils.date_utils import get_month_start_end_date
from quant_models.data_processing.features_calculation import get_equity_daily_features
from quant_models.data_processing.features_calculation import get_announcement_profitability_features
from quant_models.data_processing.features_calculation import get_halt_security_ids
from quant_models.data_processing.features_calculation import get_idx_cons_dy
from quant_models.data_processing.features_calculation import get_idx_weights
from quant_models.data_processing.features_calculation import get_sw_2nd_indust
from quant_models.utils.io_utils import write_json_file
from quant_models.utils.io_utils import load_json_file
from quant_models.data_processing.features_calculation import get_source_feature_mappings
from quant_models.data_processing.data_fetcher import DataFetcher
from quant_models.utils.helper import get_source_root
from quant_models.utils.helper import get_config
from quant_models.applications.feature_selection import feature_selection_ic
from sklearn import decomposition
from collections import defaultdict
from quant_models.model_processing.feature_preprocessing import get_sw1_indust_code

# data_fetcher_db = DataFetcherDB()
# data_fetcher_ex = DataFetcherEx()

start_date = '20160109'
end_date = '20160515'
model_name = 'linear'
config = get_config()


def _is_moonth_start_date(calendar_date=''):
    mon_start_date, mon_end_date = get_month_start_end_date(calendar_date)
    return mon_start_date.strftime('%Y%m%d') == calendar_date


def get_init_stocks(index_id='', index_date='', source=1):
    # [ticker_symbol, exchange_cd, short_name]
    # rows, desc = data_fetcher.get_idx_cons(1)
    #
    # tmp = ['{0}.{1}'.format(item[0], item[1]) for item in rows]
    # total_len = len(tmp)
    # rand_idx = np.random.randint(0, total_len, 100)
    # return [tmp[idx] for idx in rand_idx]
    # return get_idx_cons(index_id, index_date)
    return get_idx_cons_dy(security_id=index_id, index_date=index_date, source=source)


def get_trade_sec_codes(bc='', trade_date='', source=0, topk=20):
    '''
    from the score csv file trained from models to output the sec_ids with the topk scores
    :param bc: sec_id for the benchmark, e.g. '000300.XSHG'
    :param trade_date:
    :param source:
    :return:
    '''
    ret_sec_codes = get_idx_cons_dy(security_id=bc, index_date=trade_date, source=source)
    root = get_source_root()
    score_path = os.path.join(os.path.join((os.path.join(os.path.realpath(root), 'data')), 'results'),
                              'pred_score_{0}_{1}_{2}.csv'.format(model_name, '20150103', '20181231'))
    score_df = pd.read_csv(score_path)
    score_df = score_df[score_df.date == int(trade_date)].sort_values(by='score', ascending=False)
    selected_rows = []
    for _date, _secid, _score in score_df.values:
        if _secid in ret_sec_codes:
            selected_rows.append((_secid, _score))
    selected_rows.sort(key=lambda x: x[1], reverse=True)
    ret = selected_rows[:topk]
    return ret


def get_order_sec_ids(bc='', trade_date='', source=0, top_ratio=0.25, bottom_ratio=0.2, topk=20, neutralized=True,
                      topk_in=3, cached_score=True):
    '''
    from the score csv file trained from models to output the sec_ids with the topk scores
    :param bc: sec_id for the benchmark, e.g. '000300.XSHG'
    :param trade_date:
    :param source:
    :param neutralized, neutralised based on industry or not
    :return:
    '''
    next_date = datetime_delta(dt=trade_date, format='%Y%m%d', days=1)
    if cached_score:
        root = get_source_root()
        score_path = os.path.join(os.path.join((os.path.join(os.path.realpath(root), 'data')), 'results'),
                                  'pred_score_{0}_{1}_{2}.csv'.format(model_name, '20150103', '20190531'))
        score_df = pd.read_csv(score_path)
        score_df = score_df[score_df.date == trade_date]
        sec_ids = list(score_df['sec_id'])
        pred_y = list(score_df['score'])
    else:
        df = feature_selection_ic(start_date=trade_date, end_date=next_date, data_source=source,
                                  feature_types=[], train_feature=False, saved_feature=False, bc=bc,
                                  top_ratio=top_ratio, bottom_ratio=bottom_ratio)
        m = Ml_Reg_Model(model_name)
        m.load_model('stock_selection_{0}'.format(model_name))
        decom_ratio = float(config['defaults']['decom_ratio'])
        n_cols = df.shape[1]
        pca = decomposition.PCA(n_components=int(n_cols * decom_ratio))
        col_names = list(df.columns)
        col_names.remove('TRADE_DATE')
        col_names.remove('SECURITY_ID')
        col_names.remove('LABEL')
        train_X = df[col_names].values
        train_X = pca.fit_transform(train_X)
        sec_ids = list(df['SECURITY_ID'])
        pred_y = m.predict(train_X)

    rows = []
    for idx, _y in enumerate(pred_y):
        rows.append([sec_ids[idx], _y])
        rows.sort(key=lambda x: x[1], reverse=True)
    if not neutralized:
        return dict(zip([item[0] for item in rows[:topk]], [1.0 / topk] * topk))
    else:
        idx_weights = get_idx_weights(security_id=bc, index_date=trade_date, source=0)
        # sw_industry = get_sw_2nd_indust(security_ids=sec_ids, source=0)
        sw_industry = get_sw1_indust_code(sec_ids=sec_ids, trade_date=trade_date)
        in2secs = defaultdict(list)
        in2weights = dict()
        for sec_id, in1 in sw_industry.items():
            _lst = in2secs.get(in1) or []
            _lst.append(sec_id)
            in2secs.update({in1: _lst})
            _w = in2weights.get(in1) or 0.0
            _w += (idx_weights.get(sec_id) or 0.0)
            in2weights.update({in1: _w})
        sec2score = dict(zip([item[0] for item in rows], [item[1] for item in rows]))
        ret = {}
        total_weight = sum(list(in2weights.values()))
        for _in1, _sec_ids in in2secs.items():
            _sec2score = [(item, sec2score.get(item)) for item in _sec_ids]
            _sec2score.sort(key=lambda x: x[1], reverse=True)
            _in_weight = in2weights.get(_in1) / total_weight
            _ret_sec_ids = [item[0] for item in _sec2score[:topk_in]]
            ret.update(dict(zip(_ret_sec_ids, [_in_weight / topk_in] * len(_ret_sec_ids))))
    return ret


def get_sec_codes_old(select_num=500, trade_date='20181101', start_date='', end_date='', ret_features=None):
    # industry_info = ts.get_industry_classified()
    # ICSRS,TICKER_SYMBOL,EXCHANGE_CD
    logger.info('start cal sec_codes from {0} to {1}'.format(start_date, end_date))
    data_fetcher_db = DataFetcher(source=0).get_data_fetcher_obj()
    rows, desc = data_fetcher_db.get_industry_info()
    industry_set = set([item[0] for item in rows])
    industry_num = len(industry_set)
    num_each_industry = int(select_num / industry_num) or 1
    code_lst = get_init_stocks('000300.XSHG', trade_date)
    indus_scores = defaultdict(list)
    m = Ml_Reg_Model(model_name)
    m.load_model('stock_selection_{0}'.format(model_name))
    logger.info('start processing rows')
    features = get_source_feature_mappings(model_name=None, train_feature=False)
    ret_features = ret_features or get_feature(security_ids=code_lst, features=features, start_date=start_date,
                                               end_date=end_date, source=2)
    stockid_lst = [item.split('.')[0] for item in code_lst]
    # ret_prob = get_announcement_profitability_features(stock_ids=stockid_lst, start_date=start_date,
    #                                                    end_date=end_date)
    for item in rows:
        try:
            indus_id, ticker_symbol, exchange_cd = item
            sec_code = '{0}.{1}'.format(ticker_symbol, exchange_cd)
            logger.debug('processing sec_code:{0}'.format(sec_code))
            if sec_code in code_lst:
                # features = get_feature(ticker_symbol)
                try:
                    ret_feature_val = ret_features.get('{0}.{1}'.format(ticker_symbol, exchange_cd)).get(
                        int(trade_date))
                    # if ticker_symbol in ret_prob:
                    #     tmp_dict = ret_prob.get(ticker_symbol)
                    #     if tmp_dict and trade_date in tmp_dict:
                    #         ret_feature_val.update({'SENTIMENT': tmp_dict.get(trade_date)})
                    #     else:
                    #         ret_feature_val.update({'SENTIMENT': 0})
                    score = m.predict(list(ret_feature_val.values()))[0]
                    logger.debug('predict score is:{0}'.format(score))
                except Exception as ex:
                    logger.warn(
                        'Feature value missing for sec_code:{0} and trade_date:{1} with error:{2}'.format(sec_code,
                                                                                                          trade_date,
                                                                                                          ex))
                    score = 0.0
                indus_scores[indus_id].append((sec_code, score))
            else:
                logger.warn("sec_code:{0} not in init code lst".format(sec_code))
        except Exception as ex:
            raise ValueError
    logger.info('finish processing rows')
    ret_codes = []
    for indus_id, val in indus_scores.items():
        val.sort(key=operator.itemgetter(1))
        ret_codes.extend([item[0] for item in val[:num_each_industry]])
    # sorted_ret = industry_info.sort(['score'], ascending=False).groupby('c_name').head(num_each_industry)
    return ret_codes


def init(context):
    context.holdings = []
    # code_lst = get_init_stocks('000300.XSHG', context.now.strftime('%Y%m%d'))
    # features = get_source_feature_mappings(model_name=None, train_feature=False)
    # if _is_moonth_start_date(context.now.strftime('%Y%m%d')):
    #     ret_features = get_feature(security_ids=code_lst, features=features, start_date=start_date,
    #                                end_date=end_date, source=2)
    #     context.features = ret_features
    # else:
    #     context.features = None


def before_trading(context):
    pass
    # logger.debug('begin trading')
    # trade_date = context.now.strftime('%Y%m%d')
    # end_date = datetime_delta(trade_date, '%Y%m%d', days=1)
    # if _is_moonth_start_date(trade_date):
    #     context.sec_codes = get_sec_codes(30, trade_date=context.now.strftime('%Y%m%d'), start_date=trade_date,
    #                                       end_date=end_date, ret_features=context.features)
    # else:
    #     context.sec_codes = []
    # root = get_source_root()
    # select_stock_path = os.path.join(os.path.realpath(root), 'conf', 'selected_stocks_{0}.json'.format(model_name))
    # selected_stocks = load_json_file(
    #     select_stock_path) or {}
    # selected_stocks.update({trade_date: context.sec_codes})
    # mon_key = trade_date[-2:]
    #
    # write_json_file(select_stock_path, selected_stocks)
    # logger.debug('complete before trading')
    # # update_universe(context.sec_codes)


def handle_bar(context, bar_dict):
    logger.debug('call handle_bar')
    trade_date = context.now.strftime('%Y%m%d')
    # if _is_moonth_start_date(trade_date):
    if True:
        logger.info('order for month start date:{0}'.format(trade_date))
        # _ = get_trade_sec_codes(bc=__config__.get('base').get('benchmark'), trade_date=trade_date,
        #                         source=0, topk=20)
        pos_and_weights = get_order_sec_ids(bc=__config__.get('base').get("benchmark"), trade_date=trade_date, source=0,
                                            top_ratio=0.25,
                                            bottom_ratio=0.2, topk=20, cached_score=False)
        if pos_and_weights:
            init_sec_codes = list(pos_and_weights.keys())
            halt_sec_ids = get_halt_security_ids(init_sec_codes, trade_date)
            target_positions = list(set(init_sec_codes) - set(halt_sec_ids))
            if target_positions:
                curr_positions = list(context.portfolio.stock_account.positions.keys())
                sell_positions = set(curr_positions) - set(target_positions)
                for s in sell_positions:
                    order_target_percent(s, 0.0)
                # each_percent = float(0.8 / len(target_positions))
                for s in target_positions:
                    try:
                        # order_book_id = '{0}.XSHE'.format(s)
                        order_book_id = s.replace('_', '.')
                        # logger.debug('Order book is:{0} for percent:{1}'.format(order_book_id, each_percent))
                        order_target_percent(order_book_id, pos_and_weights.get(s))
                        # if order_book_id not in halt_sec_ids:
                        #     order_target_percent(order_book_id, each_percent)
                    except Exception as ex:
                        pass
            else:
                logger.info('No selected stocks')
        else:
            logger.debug('not month start, skip')


def after_trading(context):
    port_lst = list(context.portfolio.stock_account.positions.values())
    for item in port_lst:
        context.holdings.append(
            [context.now, item.order_book_id, item.quantity, item.avg_price, item.market_value,
             item.pnl])
    print(context.holdings)
    df = pd.DataFrame(context.holdings, columns=['date', 'sec_id', 'quantity', 'avg_price', 'market_value', 'pnl'])
    df.to_csv('portfolio_{0}_{1}.csv'.format(__config__.get("base").get('start_date'),
                                             __config__.get('base').get('end_date')),
              index=None)


__config__ = {
    "base": {
        "start_date": "2015-01-03",
        "end_date": "2018-12-31",
        "frequency": "1d",
        "matching_type": "current_bar",
        "data_bundle_path": "rqdata/bundle",
        "benchmark": "000300.XSHG",
        "accounts": {
            "stock": 10000000
        }
    },
    "extra": {
        "log_level": "error",
        "show": True,
    },
    "mod": {
        "sys_progress": {
            "enabled": True,
            "show": True,
        },
    },
}

if __name__ == '__main__':
    ret = get_order_sec_ids(bc='000300.XSHG', trade_date='20190618', source=0)
    print(ret)
