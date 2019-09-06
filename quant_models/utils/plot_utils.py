# -*- coding: utf-8 -*-
# @time    : 2018/9/10 17:15
# @author  : huangyu10@cmschina.com.cn
# @file    : plot_utils.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2D(values=[], styles=[],  x_label='', y_label='', saved_path=None,
            title='', legends=[]):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for idx, value in enumerate(values):
        x_values, y_values = value
        if styles[idx] == 'plot':
            plt.plot(x_values, y_values, label=u'', linestyle='-')
        elif styles[idx] == 'bar':
            plt.bar(x_values, y_values, color='b', alpha=0.65, label=u'')
        elif styles[idx] == 'scatter':
            plt.scatter(x_values, y_values, label=u'', linestyle='-')
    plt.gcf().autofmt_xdate()
    # plt.legend([x_legend, y_legend])
    plt.legend(legends)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if saved_path:
        plt.savefig(saved_path)
    else:
        plt.show()


def plot_3D(values=[], legends=[], labels=[], saved_path=None, title=''):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    for idx, value in enumerate(values):
        x_values, y_values, z_values = value
        ax.scatter(x_values, y_values, z_values, label=legends[idx])
    plt.title(title)
    plt.legend(loc='upper right')
    if saved_path:
        plt.savefig(saved_path)
    else:
        plt.show()


if __name__ == '__main__':
    import numpy as np
    x = list(range(10))
    y = [item*2 for item in x]
    plt.plot(y, label=u'', linestyle='-')
    plt.xticks([])
    plt.show()
    # plot_2D([[[1, 2], [2, 4]], [[1, 2], [2.4, 4.4]]])
