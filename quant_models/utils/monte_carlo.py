# -*- coding: utf-8 -*-
# @time      : 2019/10/25 12:32
# @author    : rpyxqi@gmail.com
# @file      : monte_carlo.py

import numpy as np
from WindPy import w
import pandas as pd
import matplotlib.pyplot as plt
import math
import pprint

def get_sigma(prices=[]):
    returns = []
    num_prices = len(prices)
    for idx in range(1, num_prices):
        returns.append(math.log(prices[idx] / prices[idx - 1]))
    return np.array(returns).std()


# S_T = S_0 exp((r-0.5*sigma^2) *T + sigma*sqrt(T) z)
def func1(S0=27.64, r=0, sigma=0.18, I=250, T=1):
    # S0 = 100
    # r = 0.05
    # sigma = 0.25
    # T = 2.0
    # I = 10000
    ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.standard_normal(I))
    # plt.hist(ST1, bins=50)
    # plt.xlabel('price')
    # plt.ylabel('frequency')
    # plt.show()
    return ST1


def func2():
    S0 = 100
    r = 0.05
    sigma = 0.25
    T = 2.0
    I = 10000
    # ST1 = S0*np.exp((r - 0.5*sigma**2)*T+sigma*np.sqrt(T)*np.random.standard_normal(I))
    # plt.hist(ST1,bins = 50)
    # plt.xlabel('price')
    # plt.ylabel('ferquency')

    M = 30
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    print(S[0])
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(I))
    plt.hist(S[-1], bins=50)
    plt.xlabel('price')
    plt.ylabel('frequency')
    plt.show()
    plt.hist(S[:, :], lw=1.5)
    plt.xlabel('time')
    plt.ylabel('price')
    # plt.show()


if __name__ == '__main__':
    # func1()
    w.start()
    # w.wsd(sec_code, "close,volume,turn,open,chg,pct_chg,total_shares", t_days[-pe
    ret = w.wsd('002180.SZ', 'pct_chg', '20190802', '20191101')
    prices = list(np.array(ret.Data[0]))
    log_returns = []
    print(ret.Data[0])
    print(np.array(ret.Data[0]).std()/math.sqrt(60))

    #30天，0.54
    # mock_prices = []
    # for i in range(100):
    #     ret = func1(S0=27.64, r=0, sigma=0.18, I=1000, T=0.15)
    #     mock_prices.append(np.array(ret).mean())
    #     # pprint.pprint(np.array(ret).mean())
    # arr_p = np.array(mock_prices)
    # print(arr_p.max(), arr_p.min(), arr_p.mean())


    # func2()
