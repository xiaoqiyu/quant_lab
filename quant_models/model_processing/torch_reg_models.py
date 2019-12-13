# -*- coding: utf-8 -*-
# @time      : 2019/12/11 18:45
# @author    : rpyxqi@gmail.com
# @file      : torch_reg_models.py


import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import uqer
from uqer import DataAPI

torch.manual_seed(1)
# HYPER Parameters
TIME_STEP = 10  # rnn time step/image height
INPUT_SIZE = 2
LR = 0.2
DOENLOAD_MINST = False
BATCH_SIZE = 10
HIDDEN_SIZE = 32
NUM_LAYER = 2
uqer_client = uqer.Client(token="26356c6121e2766186977ec49253bf1ec4550ee901c983d9a9bff32f59e6a6fb")


def get_features(security_id=u"300634.XSHE", date='20191122'):
    ticker, exchange_cd = security_id.split('.')
    df = DataAPI.MktTicksHistOneDayGet(securityID=security_id, date=date, startSecOffset="", endSecOffset="",
                                       field=u"", pandas="1")
    df_min = DataAPI.SHSZBarHistOneDayGet(tradeDate=date, exchangeCD=exchange_cd, ticker=ticker, unit="5",
                                          startTime=u"", endTime=u"", field=u"", pandas="1")
    datatimes = list(df['dataTime'])
    total_vol = list(df['value'])[-1]
    data_min = ['{0}:{1}'.format(item.split(':')[0], item.split(':')[1]) for item in datatimes]
    df['dataMin'] = data_min
    df['avgPrice'] = df['value'] / df['volume']
    df['amplitude'] = (df['highPrice'] - df['lowPrice']) / df['lastPrice']
    df['spread'] = df['askPrice1'] - df['bidPrice1']
    df['openDiff'] = (df['openPrice'] - df['prevClosePrice']) / df['prevClosePrice']
    df['trackError'] = (df['lastPrice'] - df['avgPrice']) / df['avgPrice']
    df['askTrackError1'] = (df['askPrice1'] - df['avgPrice']) / df['avgPrice']
    df['bidTrackError1'] = (df['bidPrice1'] - df['avgPrice']) / df['avgPrice']
    df['totalAskVolume'] = df['askVolume1'] + df['askVolume2'] + df['askVolume3'] + df['askVolume4'] + df['askVolume5']
    df['totalBidVolume'] = df['bidVolume1'] + df['bidVolume2'] + df['bidVolume3'] + df['bidVolume4'] + df['bidVolume5']
    df['volumeImbalance1'] = (df['askVolume1'] - df['bidVolume1']) / (df['askVolume1'] + df['bidVolume1'])
    df['volumeImbalanceTotal'] = (df['totalAskVolume'] - df['totalBidVolume']) / (
            df['totalAskVolume'] + df['totalBidVolume'])
    df['volumePerDeal'] = df['volume'] / df['deal']
    df['volumeRatio'] = df['volume'] / total_vol
    int(list(set(df['dataMin']))[0].split(':')[1]) % 5
    print(df.shape)
    print(df.columns)

    min_vwap = list(df_min['vwap'])
    min_vwap.insert(0, min_vwap[0])
    df_min['ret'] = (df_min['vwap'] / min_vwap[:-1] - 1) * 100
    dict_min_ret = dict(zip(df_min['barTime'], df_min['ret']))
    # TODO add features of tick and min
    columns = list(df.columns)
    columns.remove('dataDate')
    columns.remove('exchangeCD')
    columns.remove('ticker')
    # columns.remove('dataTime')
    # columns.remove('dataMin')
    df.sort_values(by='dataTime', ascending=True, inplace=True)
    data_min = list(df['dataMin'])
    df = df[columns]
    rows = list(df.values)
    train_x = []
    _start, _end = 0, 0
    n_row = len(rows)
    for idx, val in enumerate(rows):
        hh, mm = data_min[idx].split(':')
        if int(hh) == 9 and int(mm) <= 30:
            continue
        _start = idx
        if idx == n_row-1:
            pass
        else:
            hh_, mm_ = data_min[idx+1].split(':')
            if int(mm) % 5 == 0 and int(mm_) != int(mm):
                print(data_min[_start], data_min[idx])
                train_x.append(rows[_start: idx])
                _start = idx
    return train_x



class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size=INPUT_SIZE,  # 输入的特征维度
                                hidden_size=HIDDEN_SIZE,  # rnn hidden layer unit
                                num_layers=NUM_LAYER,  # 有几层RNN layers
                                batch_first=True)  # input & output 会是以batch size 为第一维度的特征值
        # e.g. (batch, seq_len, input_size)
        self.out = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, h_state):  # 因为hidden state是连续的，所以我们要一直传递这个state
        # x(batch, seq_len/time_step, input_size)
        # h_state(n_layers, batch, hidden_size)
        # r_out(batch, time_step, output_size)
        r_out, h_state = self.rnn(x, h_state)  # h_state 也要作为RNN的一个输入
        outs = self.out(h_state[-1, :, :])
        return r_out, outs, h_state


def rnn_reg_training():
    rnn = RNN()
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = torch.nn.MSELoss()

    h_state = None  # 要使用初始hidden state, 可以设成None
    plt.ion()
    plt.show()
    for step in range(100):
        # FIXME generate random test case
        x_np = np.random.random(BATCH_SIZE * TIME_STEP * INPUT_SIZE).reshape(BATCH_SIZE, TIME_STEP, INPUT_SIZE)
        y_np = np.random.random(BATCH_SIZE).reshape(BATCH_SIZE, 1)

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        prediction, outputs, h_state = rnn(x, h_state)  # rnn对于每一个step的prediction, 还有最后一个step的h_state
        h_state = h_state.data  # 要把h_state 重新包装一下才能放入下一个iteration,不然会报错
        loss = loss_func(outputs, y)
        # loss = loss_func(prediction, y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backprogation, compute gradients
        optimizer.step()  # apply gradients
        if step % 5 == 0:
            plt.cla()
            plt.scatter(range(x.shape[0]), y[:, 0].data.numpy())
            plt.plot(range(x.shape[0]), outputs[:, 0].data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(1.5)


if __name__ == '__main__':
    # rnn_reg_training()
    get_features()
