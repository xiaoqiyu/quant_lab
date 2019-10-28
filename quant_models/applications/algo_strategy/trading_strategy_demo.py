# -*- coding: utf-8 -*-
# @time      : 2019/10/28 20:24
# @author    : rpyxqi@gmail.com
# @file      : trading_strategy_demo.py

from quant_models.applications.algo_strategy.trading_strategy import *


def main():
    ret = get_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="300694.SZ",
                       end_date="2019-10-22", period=100, target_vol=2150000 - 642000 - 664210, target_period=None,
                       price_ratio=0.98, update=True)
    print(ret)

    ret = get_trade_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="300694.SZ",
                             end_date="2019-10-21", period=100, target_vol=1508000, target_period=None,
                             price_ratio=0.98)
    print(ret)


if __name__ == '__main__':
    main()
