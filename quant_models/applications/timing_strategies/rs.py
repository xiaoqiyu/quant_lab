# -*- coding: utf-8 -*-
# @Time      : 2019/11/14 13:55
# @Author    : lxfei1220@163.com


import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from WindPy import w

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


class RS(object):
    """
    cal rs, can not cal single rs value
    """
    def __init__(self, high, low, window):
        self.high = high
        self.low = low
        self.window = window

    def _cal_rs(self, zscore_window=20):
        """
        cal rs value
        :return:
        """
        high_length = len(self.high)
        low_length = len(self.low)

        if high_length != low_length:
            raise Exception("length of high and low must be equal")

        if high_length < self.window + zscore_window - 1:
            raise Exception("length of high and low are too short, rs_window and zscore_window are too long")

        if not isinstance(self.high, (pd.Series, pd.DataFrame)) and \
                not isinstance(self.high, (pd.Series, pd.DataFrame)):
            self.high = pd.Series(self.high)
            self.low = pd.Series(self.low)

        model = LinearRegression()
        beta = []
        r_square = []
        for i in range(self.window, 1 + high_length):
            high_window = self.high[i - self.window: i]
            low_window = self.low[i - self.window: i]
            model.fit(low_window.values.reshape(-1, 1), high_window.values.reshape(-1, 1))
            beta.append(model.coef_[0][0])
            r_square.append(model.score(low_window.values.reshape(-1, 1), high_window.values.reshape(-1, 1)))

        beta = pd.Series(beta, index=self.high.index[self.window - 1:])
        beta_stand = beta.rolling(window=zscore_window).apply(lambda x: (x[-1] - x.mean()) / x.std(ddof=1))
        rs_value = beta_stand * np.array(r_square)
        rs_value = pd.Series(rs_value, index=self.high.index[self.window + zscore_window - 2:])
        rs_value_right = rs_value * beta
        return beta, beta_stand, rs_value, rs_value_right

    def rs(self, zscore_window):
        return self._cal_rs(zscore_window)


if __name__ == "__main__":
    # np.random.seed(42)
    # a = np.arange(100)
    # b = np.arange(100) * 2 + np.random.rand(1, 100) * 10 - 5
    # b = b[0]
    # c = RS(a, b, 20).rs(10)
    # c.hist()
    # plt.show()

    w.start()
    data = w.wsd("000001.SH", "open,high,low,close,amt", "2015-10-26", "2019-11-24", "PriceAdj=F")
    w.stop()

    data = pd.DataFrame(data.Data, index=data.Fields, columns=data.Times).T
    beta, beta_stand, rs_value, rs_value_right = RS(data["LOW"], data["HIGH"], window=20).rs(zscore_window=250)

    """
    # TODO plot figure
    f1 = plt.figure(figsize=(20, 10))
    # sub_figure_1
    ax1 = f1.add_subplot(211)
    ax1_1 = ax1.twinx()
    data["CLOSE"].plot(ax=ax1, ylim=[2000,3700], color="b")
    rs_value.plot(ax=ax1_1, ylim=[-3, 10], color="r", alpha=0.7)
    ax1.legend(ax1.lines + ax1_1.lines, ["上证综指", "rs指数"])
    ax1.set_ylabel("上证综指")
    ax1_1.set_ylabel("rs指数")
    # sub_figure_2
    ax2 = f1.add_subplot(212)
    rs_value.hist(ax=ax2, bins=100, color="r", alpha=0.7)
    ax2.legend(["rs指数频率分布直方图",])
    plt.show()

    # TODO plot figure
    f2 = plt.figure(figsize=(20, 10))
    # sub_figure_1
    ax1 = f2.add_subplot(211)
    ax1_1 = ax1.twinx()
    data["CLOSE"].plot(ax=ax1, ylim=[2000,3700], color="b")
    rs_value_right.plot(ax=ax1_1, ylim=[-3, 10], color="r", alpha=0.7)
    ax1.legend(ax1.lines + ax1_1.lines, ["上证综指", "rs指数"])
    ax1.set_ylabel("上证综指")
    ax1_1.set_ylabel("rs指数")
    # sub_figure_2
    ax2 = f2.add_subplot(212)
    rs_value_right.hist(ax=ax2, bins=100, color="r", alpha=0.7)
    ax2.legend(["rs指数频率分布直方图",])
    plt.show()
    """

    print("success")
