# -*- coding: utf-8 -*-
# @time      : 2018/11/7 10:46
# @author    : rpyxqi@gmail.com
# @file      : data_fetcher.py


from quant_models.utils.decorators import parallel_pool
from quant_models.utils.io_utils import write_json_file
from quant_models.utils.logger import Logger
from quant_models.data_processing.data_fetcher_cache import DataFetcherCache
from quant_models.data_processing.data_fetcher_db import DataFetcherDB
from quant_models.data_processing.date_fetcher_api import DataFetcherAPI

logger = Logger(log_level='INFO').get_log()


class DataFetcher(object):
    def __init__(self, source=0):
        '''
        :param source: int; 0 for db, 1 for cache, 2 for api
        '''
        self._source = source
        self._db_obj = DataFetcherDB()
        self._cache_obj = DataFetcherCache()
        self._db_api = DataFetcherAPI()

    def get_data_fetcher_obj(self, source=0):
        return {0: self._db_obj,
                1: self._cache_obj,
                2: self._db_api}.get(source or self._source)


@parallel_pool
def get_factor_parallel(factors, security_ids=(), start_date=None, end_date=None, df_obj=None):
    factor_type, fields = factors
    print('input factor is:{0}'.format(factors))
    rows, desc = df_obj.get_equ_factor(factor_type, security_ids, fields, start_date, end_date)
    print('query result in parallel func', rows, desc)
    return [rows, desc]


if __name__ == '__main__':
    # TODO REMOVE THE TESTING CODES TO UNITTEST
    import pprint

    df = DataFetcher(source=0)
    # ret = df.get_equ_factor(factor_type='growth', security_ids=['601928.XSHG'], start_date=20160103, end_date=20170301,
    #                         source='file')
    # print(ret)
    # halt_info = df.get_halt_info(security_ids=['100567.XSHG', '100096.XSHG'])
    # pprint.pprint(halt_info)
    # ret = get_latest_weight()
    # pprint.pprint(len(ret))

    # rows, desc = df.get_idx_cons(1)
    # print(len(rows))
    # rows, desc = df.get_equ_factor(fields=['ACD6', 'ACD20'], factor_type='ma', security_ids=['000622.XSHE'],
    #                                start_date=20181101, end_date=20181101)
    # print(rows)
    # rows, desc = df.get_mkt_equd(fields=['PRE_CLOSE_PRICE', 'CLOSE_PRICE', 'OPEN_PRICE'], security_ids=['000001.XSHE'],
    #                              start_date='20181101', end_date='20181102')
    # pprint.pprint(rows)
    # rows, desc = df.get_annoucement_profitability(stock_ids=['300100', '601717'], start_date='20170101',
    #                                               end_date='20181102')
    # pprint.pprint(rows)
    # print(desc)
    # rows, desc = df.get_mkt_idxd(fields=['PRE_CLOSE_INDEX', 'CLOSE_INDEX', 'OPEN_INDEX'], security_ids=['000001.XSHG'],
    #                              start_date='20181101', end_date='20181102')
    # pprint.pprint(rows)
    #
    features = {'growth': [],
                'vs': [],
                'return': [],
                'ma': [],
                'obos': [],
                'power': [],
                'trend': [],
                'volume': [],
                'psi': [],
                'pq': [],
                'sc': [],
                'cf': [],
                'oc': [],
                'af': [],
                'derive': [],
                }
    # f_mappings = {}
    # excluded_fiels = set(['ID', 'SECURITY_ID_INT', 'SECURITY_ID', 'TRADE_DATE', 'TICKER_SYMBOL', 'CREATE_TIME',
    #                       'UPDATE_TIME'])
    # for ftype, fields in features.items():
    #     rows, desc = df.get_equ_factor(fields=fields, factor_type=ftype, security_ids=['000622.XSHE'],
    #                                    start_date=20181101, end_date=20181102)
    #     lst = list(set(desc) - excluded_fiels)
    #     f_mappings.update({ftype: lst})
    # import pprint
    #
    # pprint.pprint(f_mappings)
    #
    # write_json_file('E:\pycharm\\algo_trading\quant_models\quant_models\conf\\feature_mapping.json', f_mappings)
    # ret = get_factor_parallel([['cf', ['OperCashGrowRate']], ['growth', ['NetAssetGrowRate']]],
    #                           security_ids=['000622.XSHE'],
    #                           start_date=20181101, end_date=20181102, df_obj=df)
    # print('results is', ret)
    # rows, desc = df.get_idx_cons(1782)
    # print(len(rows))

    # dfb = DataFetcherDB()
    # rows, desc = dfb.get_data_by_sql("SELECT FULL_NAME,JSON_CONTENT,CREATE_DT FROM REPORT_CONTENT")
    # df_report = pd.DataFrame(rows, columns=desc)
    # df_report.to_csv("E:\pycharm\\algo_trading\quant_models\quant_models\data\originals\\report_content.csv")

    # rows, desc = df._df_obj.get_data_by_sql(
    #     "SELECT TRADE_DATE,VEMA5,VEMA10,VR FROM EQU_FACTOR_VOLUME WHERE TICKER_SYMBOL='001979' ORDER BY trade_date")
    #
    # vema5 = [item[1] for item in rows]
    # vema10 = [item[2] for item in rows]
    # dates = [item[0] for item in rows]
    # import matplotlib.pyplot as plt
    #
    # plt.plot(vema5, 'r')
    # plt.plot(vema10, 'b')
    #
    # plt.show()

    # import uqer
    # from uqer import DataAPI
    # from quant_models.utils.date_utils import datetime_delta
    #
    # c = uqer.Client(token="cae4e8fdd64a6cb9c68e9014ab04fdd823da6c41a77417ce6c8dbdf31db35541")
    # ll = len(rows)
    # print('len is:',ll)
    # cnt = 0
    # for item in rows:
    #     print('{0} out of {1}'.format(cnt,ll))
    #     cnt+=1
    #     ret = DataAPI.AnnouncementGet(ticker=item[0],reportID=u"",beginDate=u"20160101",endDate=u"20181226",field=u"",pandas="1")
    #     ret.to_csv(
    #         "E:\pycharm\\algo_trading\quant_models\quant_models\data\originals\\announcement\\contents_{0}.csv".format(item[0]))

    # ret = df.get_data_fetcher_obj().get_halt_info(['300779.XSHE'])
    # pprint.pprint(ret)
    # ret = df.get_data_fetcher_obj().get_indust_stats()
    # pprint.pprint(ret)
    # rows, cols = df.get_data_fetcher_obj().get_sw_indust()
    # print(rows)
    # rows, cols = df.get_data_fetcher_obj().get_security_codes()
    # print(rows, cols)

    # rows, cols = df.get_data_fetcher_obj().get_idx_cons_jy(idx_security_id='000300')
    # print(rows, cols)

    # rows, cols = df.get_data_fetcher_obj()._get_sec_main(inner_codes=[3, 6, 26])
    # print(rows, cols)

    # _rows, cols = df.get_data_fetcher_obj().get_sw_idx_codes_jy()
    #
    # rows, cols = df.get_data_fetcher_obj().get_indust_mkt_jy(idx_codes=[item[0] for item in _rows],
    #                                                          start_date='20190701')
    # pprint.pprint(rows)

    row, cols = df.get_data_fetcher_obj().get_sw_2nd()
    print(row)
