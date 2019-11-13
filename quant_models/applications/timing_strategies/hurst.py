# -*- coding: utf-8 -*-
# @Time      : 2019/11/12 13:55
# @Author    : lxfei1220@163.com


from collections import Iterable
import logging
import warnings
from pprint import pprint
import numpy as np
from WindPy import w
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


class HurstExponent(object):

    def __init__(self, ts, *args):
        self.ts = ts
        self.window = args[0] if args else ()
        self.flag = 1 if args else 0

    def _refactor(self):
        """
        factorization
        :return: list, [(factor0_0, factor0_1), (factor1_0, factor1_1), (factor2_0, factor2_1), ...]
        """
        loop_end_value = self.window // 2
        factor_combination = []
        for i in range(2, loop_end_value):
            i_factor = self.window % i
            if i_factor == 0:
                factor_combination.append((i, self.window // i))
        return factor_combination

    def _force_refactor(self):
        """
        force factorization
        :return: list, [(factor0_0, factor0_1), (factor1_0, factor1_1), (factor2_0, factor2_1), ...]
        """
        loop_end_value = self.window // 2
        factor_combination = [(i, self.window // i, self.window % i) for i in range(2, loop_end_value)]
        return factor_combination

    def _generate_subsequence(self, ts_window):
        """
        generate subsequence
        :return: list, [[series0_0, series0_1], [series1_0, series1_1, series1_2], ...]
        """
        factor_combination = self._refactor()
        factor_combination_length = len(factor_combination)
        subsequence_list = []
        if factor_combination_length < 5:
            logging.info("length of series can not be refactored, try to force refactor")
            factor_combination = self._force_refactor()
            for ix, iy, iz in factor_combination:
                subsequence = [ts_window[z + x * iy: z + (1 + x) * iy] for z in range(1 + iz) for x in range(ix)]
                subsequence_list.append(subsequence)
        else:
            for ix, iy in factor_combination:
                subsequence = [ts_window[x * iy: (1 + x) * iy] for x in range(ix)]
                subsequence_list.append(subsequence)
        return subsequence_list

    @staticmethod
    def cal_rs(ts_window_subsequence):
        """
        calculate in subsequence
        :return: float
        """
        ts_cumsum = np.cumsum(ts_window_subsequence - np.mean(ts_window_subsequence))
        ts_range = max(ts_cumsum) - min(ts_cumsum)
        ts_std = np.std(ts_window_subsequence)
        ts_rs = ts_range / ts_std
        return ts_rs

    def _cal_hurst(self):
        """
        calculate hurst exponent
        :return: moving hurst: list/series, single hurst: float
        """
        if not isinstance(self.ts, Iterable):
            raise Exception("input value is not iterable")

        if not isinstance(self.ts, (pd.Series, pd.DataFrame)):
            self.ts = pd.Series(self.ts)
        self.ts = self.ts[self.ts.notnull()]

        if not self.window:
            self.window = len(self.ts)

        hurst_exponent = []
        for i in range(self.window, 1+len(self.ts)):
            ts_window = self.ts[i - self.window:i]
            subsequence_list = self._generate_subsequence(ts_window)
            rs_list = []
            num_list = []
            for subsequence in subsequence_list:
                subsequence_rs = np.log(np.mean([self.cal_rs(x) for x in subsequence]))
                subsequence_length = np.log(len(subsequence[0]))
                rs_list.append(subsequence_rs)
                num_list.append(subsequence_length)
            hurst_exponent.append(np.polyfit(num_list, rs_list, 1)[0])

        if self.flag:
            return pd.Series(hurst_exponent, index=self.ts.index[self.window-1:])
        else:
            return hurst_exponent[0]

    @property
    def hurst_exponent(self):
        return self._cal_hurst()


if __name__ == "__main__":

    w.start()
    data = w.wsd("000300.SH", "close", "2017-01-01", "2018-12-31", "PriceAdj=F")
    w.stop()

    close = pd.Series(data.Data[0], index=data.Times)
    ret = close.pct_change()

    hurst_test = HurstExponent(ret, 120).hurst_exponent
    hurst_test.plot()
    plt.show()

    pprint(hurst_test)

    # a = range(250)
    # b = HurstExponent(a, 120).hurst_exponent
    # print(b)
