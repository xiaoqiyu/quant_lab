# -*- coding: utf-8 -*-
# @time      : 2019/10/16 11:29
# @author    : rpyxqi@gmail.com
# @file      : trading_strategy.py

from WindPy import w
import talib as ta
import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt

w.start()


def get_trade_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="002299.SZ", start_date="",
                       end_date="2019-10-15", period=100, target_vol=2150000, target_period=20, price_ratio=0.97):
    _t_days = w.tdays("2017-01-14", end_date)
    t_days = [item.strftime('%Y-%m-%d') for item in _t_days.Data[0]]

    ret = w.wsd(sec_code, "close,volume,turn,open,chg,pct_chg,total_shares", t_days[-period], end_date, "")
    # print(ret.Data)
    vols = ret.Data[1]
    turns = ret.Data[2]
    total_shares = ret.Data[-1]
    price_ma5 = np.array(ret.Data[0][-5:]).mean() * price_ratio
    price_prev = ret.Data[0][-1] * price_ratio
    # print("price line is :{0}".format(ma5))
    ma_vols = list(ta.SMA(np.array(list(vols), dtype=float), timeperiod=5))
    # ma_turns = list(ta.SMA(np.array(list(turns), dtype=float), timeperiod=5))

    idx = -1
    total_vol = 0.0
    ret_vols = []

    # compute target vol
    target_vol = target_vol or target_ratio * total_shares[-1]

    # update the participant_rate according to the target complete period
    if target_period:
        participant_rate = target_vol / sum(ma_vols[-target_period:])

    while total_vol < target_vol:
        if target_vol - total_vol <= 100:
            ret_vols.append(100)
            break
        try:
            _vol = int(ma_vols[idx] * participant_rate / 100) * 100
        except:
            break
        ret_vols.append(_vol)
        total_vol += _vol
        idx -= 1
    return ret_vols, [price_prev, price_ma5]


def get_schedule(**kwargs):
    sec_code = kwargs.get('sec_code')
    end_date = kwargs.get('end_date')
    if not sec_code or not end_date:
        raise ValueError('sec_code and end_date should not be empty')
    participant_rate = kwargs.get('participant_rate') or 0.15
    target_ratio = kwargs.get('target_ratio') or 0.01
    period = kwargs.get('period') or 100
    target_vol = kwargs.get('target_vol') or 8000000
    target_period = kwargs.get('target_period')
    price_ratio = kwargs.get('price_ratio') or 0.95
    update = kwargs.get('update') or False
    ret_vol, ret_price = get_trade_schedule(participant_rate=participant_rate, target_ratio=target_ratio,
                                            sec_code=sec_code, end_date=end_date, period=period, target_vol=target_vol,
                                            target_period=target_period, price_ratio=price_ratio)
    # FIXME fix the border
    print(sum(ret_vol), target_vol)
    next_date = w.tdaysoffset(1, end_date).Data[0][0].strftime('%Y-%m-%d')
    sec_code = sec_code
    vol = ret_vol[0]
    price = min(ret_price)
    df = pd.DataFrame([[sec_code, next_date, vol, price]], columns=['sec_code', 'trade_date', 'vol', 'price'])
    curr_df = pd.read_csv('data/trade_strategy.csv')
    curr_df = curr_df.append(df)
    if update:
        curr_df.to_csv('data/trade_strategy.csv', index=False)
    return df


def main():
    trade_history = pd.read_excel('data/trade_history.xlsx')
    df = trade_history['002936']
    print(df)


if __name__ == '__main__':
    # # 郑州银行
    # ret = get_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="002936.SZ",
    #                    end_date="2019-10-18", period=100, target_vol=8000000, target_period=20,
    #                    price_ratio=0.95, update=True)
    # print(ret)
    #
    # # 蠡湖股份
    # ret = get_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="300694.SZ",
    #                    end_date="2019-10-18", period=100, target_vol=2150000, target_period=None,
    #                    price_ratio=0.98, update=True)
    # print(ret)
    #
    # 郑州银行
    # ret = get_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="002936.SZ",
    #                    end_date="2019-10-22", period=100, target_vol=8000000, target_period=10,
    #                    price_ratio=0.95, update=True)
    # print(ret)
    #
    # # 蠡湖股份
    ret = get_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="300694.SZ",
                       end_date="2019-10-22", period=100, target_vol=2150000-642000-664210, target_period=None,
                       price_ratio=0.98, update=True)
    print(ret)

    # ret = get_trade_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="300694.SZ",
    #                    end_date="2019-10-21", period=100, target_vol=1508000, target_period=None,
    #                    price_ratio=0.98)
    # print(ret)

    # 索通发展，参数未设置
    # ret_vols, ret_prices = get_trade_schedule(participant_rate=0.10, target_ratio=0.01, sec_code="603612.SH",
    #                                           end_date="2019-10-23", period=100, target_vol=3402350, target_period=11,
    #                                           price_ratio=0.98)
    # print(ret_vols)
    # print(ret_prices)
    # ret = get_schedule(participant_rate=0.10, target_ratio=0.01, sec_code="603612.SH",
    #                    end_date="2019-10-22", period=100, target_vol=3402350, target_period=11,
    #                    price_ratio=0.98, update=False)
    # print(ret)



    # 欧陶康视    #交易路径模拟
    # ret_vols10, ret_prices = get_trade_schedule(participant_rate=0.10, target_ratio=0.01, sec_code="300595.SZ",
    #                                             end_date="2019-10-22", period=100, target_vol=None, target_period=None,
    #                                             price_ratio=0.98)
    # ret_vols5, ret_prices = get_trade_schedule(participant_rate=0.05, target_ratio=0.01, sec_code="300595.SZ",
    #                                            end_date="2019-10-22", period=100, target_vol=None, target_period=None,
    #                                            price_ratio=0.98)
    # ret_vols15, ret_prices = get_trade_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="300595.SZ",
    #                                             end_date="2019-10-22", period=100, target_vol=None, target_period=None,
    #                                             price_ratio=0.98)

    # print(len(ret_vols5), len(ret_vols10), len(ret_vols15))
    # print()
    # plt.plot([item/100 for item in ret_vols5])
    # plt.plot([item/100 for item in ret_vols10])
    # plt.plot([item/100 for item in ret_vols15])
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.legend(["5%市场参与比率", "10%市场参与比率", "15%市场参与比率"])
    # plt.xlabel("第i个交易日")
    # plt.ylabel("每日交易量（手）")
    # # plt.show()
