# -*- coding: utf-8 -*-
# @time      : 2019/10/16 11:29
# @author    : rpyxqi@gmail.com
# @file      : trading_strategy.py

from WindPy import w
import talib as ta
import numpy as np
import pandas as pd
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
    # participant rate testing
    # p_rates_up = []
    #     # p_rates_down = []
    #     # for target_period in [10, 20, 30, 40, 50, 60]:
    #     #     p_rates_up.append(100 * (target_vol / (sum(ma_vols[-target_period:]) * 1.05)))
    #     #     p_rates_down.append(100 * (target_vol / (sum(ma_vols[-target_period:]) * .95)))
    #     # return p_rates_up, p_rates_down
    # update the participant_rate according to the target complete period
    if target_period:
        participant_rate = target_vol / sum(ma_vols[-target_period:])
        print('updated participant rate is:{0}'.format(participant_rate))

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


def nasida():
    # 高位：10-21-11.53;10-16:10.69; 中位：6-3：9.36；5-22：9.98；低位：1-31：23.12%；11-09：18.28;2-13;08-09:
    # 中：6-4-7.45；10-21：11.53；低：1-4：
    ret_vols5, ret_prices = get_trade_schedule(participant_rate=0.1, target_ratio=0.01, sec_code="002180.SZ",
                                               end_date="2019-10-11", period=100, target_vol=None, target_period=10,
                                               price_ratio=0.98)
    ret_vols10, ret_prices = get_trade_schedule(participant_rate=0.1, target_ratio=0.01, sec_code="002180.SZ",
                                                end_date="2018-06-04", period=100, target_vol=None, target_period=22,
                                                price_ratio=0.98)
    ret_vols15, ret_prices = get_trade_schedule(participant_rate=0.08, target_ratio=0.01, sec_code="002180.SZ",
                                                end_date="2019-01-04", period=100, target_vol=None, target_period=None,
                                                price_ratio=0.98)
    ret_vols20, ret_prices = get_trade_schedule(participant_rate=0.05, target_ratio=0.01, sec_code="002180.SZ",
                                                end_date="2019-01-04", period=100, target_vol=None, target_period=22,
                                                price_ratio=0.98)
    # 8-9
    print(ret_vols5)
    print(len(ret_vols15))
    plt.plot([item / 100 for item in ret_vols5])
    plt.plot([item / 100 for item in ret_vols10])
    plt.plot([item / 100 for item in ret_vols15])
    plt.plot([item / 100 for item in ret_vols20])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend(["情形1", "情形2", "情形3-a", "情形3-b"])
    plt.xlabel("第i个交易日")
    plt.ylabel("每日交易量（手）")
    plt.title("纳思达减持1%股份交易路径模拟")
    plt.show()


if __name__ == '__main__':
    nasida()
