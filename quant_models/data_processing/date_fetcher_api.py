# -*- coding: utf-8 -*-
# @time      : 2019/1/3 12:13
# @author    : rpyxqi@gmail.com
# @file      : date_fetcher_api.py

import rqdatac
import uqer
from uqer import DataAPI
from quant_models.utils.logger import Logger

logger = Logger(log_level='DEBUG', handler='ch').get_log()


class DataFetcherAPI(object):
    def __init__(self, source=0):
        '''

        :param source: 0 for uqer, 1 for rq
        '''
        self.source = source
        if source == 0:
            self._uqer_client = uqer.Client(token="26356c6121e2766186977ec49253bf1ec4550ee901c983d9a9bff32f59e6a6fb")
        else:
            # TODO this is not supported in the forllowing data applications now
            self._rq_client = rqdatac.init('user3zszq@ricequant.com', '_admin123@qq.com')

    def get_equ_factor(self, factor_type='', security_ids=(), fields=None, start_date=None, end_date=None):
        '''
        Factor not supported by applications now
        :param factor_type:
        :param security_ids:
        :param fields:
        :param start_date:
        :param end_date:
        :return:
        '''
        pass

    def get_mkt_equd(self, security_ids=(), fields=None, start_date=None, end_date=None, asset_type='stock'):
        ret = DataAPI.MktEqudGet(secID='603612.XSHG', field='closePrice', beginDate="20181103", endDate="20181130")
        return ret

    def get_idx_cons(self, idx_id=None, ticker=None, index_date=None):
        '''

        :param idx_id: 1:上证综指； 1782：沪深300
        :return:
        '''
        if self.source == 0:
            ret = list(DataAPI.IdxConsGet(ticker=ticker, isNew=u"", intoDate=index_date,
                                          field=["consTickerSymbol", 'consExchangeCD'], pandas="1").values)
            return ['{0}.{1}'.format(item[0], item[1]) for item in ret]
        elif self.source == 1:
            # TODO to be added for rqdata when the proxy for SDK is solved
            ret = rqdatac.index_components('000016.XSHG')
            return ret

    def get_news(self, start_date='', end_date=''):
        i_date = start_date
        while i_date < end_date:
            logger.info('processing date:{0}'.format(i_date))
            ret = DataAPI.NewsInfoByTimeGet(newsBeginDate=i_date, newsEndDate=i_date)

    def get_data_theme(self):
        ret = DataAPI.TkgThemesGet(pandas="1")

    def get_theme_sec_map(self):
        ret = DataAPI.TkgThemeTickerRelGet(themeID=u"", secID=u"", ticker=u"", exchangeCD=u"", field=u"", pandas="1")

    def get_social(self, start_date='', end_date=''):
        i_date = start_date
        while i_date < end_date:
            logger.info('processing date:{0}'.format(i_date))
            ret = DataAPI.SocialDataXQByDateGet(statisticsDate=i_date, field=u"", pandas="1")

    def get_social_theme(self, start_date='', end_date=''):
        i_date = start_date
        while i_date < end_date:
            logger.info('processing date:{0}'.format(i_date))
            ret = DataAPI.SocialThemeGbByDateGet(tradeDate=i_date, field=u"", pandas="1")


if __name__ == '__main__':
    df = DataFetcherAPI(0)
    # ret = df.get_mkt_equd()
    # ret = df.get_idx_cons(ticker='000300',index_date='20190401')
    # import pandas as pd
    # df = pd.DataFrame(ret)
    # df.to_csv('stock_id.csv')
    # print(ret)
    df_idx = DataAPI.IdxConsGet(secID=u"", ticker=u"000300", isNew=u"", intoDate=u"20141231", field=u"", pandas="1")

