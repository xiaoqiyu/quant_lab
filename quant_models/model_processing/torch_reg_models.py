# -*- coding: utf-8 -*-
# @time      : 2019/12/11 18:45
# @author    : rpyxqi@gmail.com
# @file      : torch_reg_models.py


import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import utils as nn_utils
import matplotlib.pyplot as plt

import uqer
from uqer import DataAPI

torch.manual_seed(1)
# HYPER Parameters
TIME_STEP = 10  # rnn time step/image height
INPUT_SIZE = 42
LR = 0.2
DOENLOAD_MINST = False
BATCH_SIZE = 47
HIDDEN_SIZE = 32
NUM_LAYER = 2
MAX_LEN = 100
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
    columns.remove('dataTime')
    columns.remove('dataMin')
    columns.remove('shortNM')
    columns.remove('currencyCD')
    df.sort_values(by='dataTime', ascending=True, inplace=True)
    data_min = list(df['dataMin'])
    df = df[columns]
    rows = list(df.values)
    train_x = []
    train_y = []
    _start, _end = 0, 0
    n_row = len(rows)
    for idx, val in enumerate(rows):
        hh, mm = data_min[idx].split(':')
        if int(hh) == 9 and int(mm) <= 30:
            continue
            _start = idx
        if idx == n_row - 1:

            if dict_min_ret.get('{0}:{1}'.format(hh, mm)):
                train_y.append([dict_min_ret.get('{0}:{1}'.format(hh, mm))])
                train_x.append(list(rows[_start: idx]))
            _start = idx
        else:
            hh_, mm_ = data_min[idx + 1].split(':')
            if int(mm) % 5 == 0 and int(mm_) != int(mm):
                print(data_min[_start], data_min[idx])
                if dict_min_ret.get('{0}:{1}'.format(hh, mm)):
                    train_y.append([dict_min_ret.get('{0}:{1}'.format(hh, mm))])
                    train_x.append(list(rows[_start: idx + 1]))
                _start = idx + 1
    return train_x, train_y


def get_data_loader():
    train_x, train_y = get_features()
    seq_lengths = []
    for _i, item in enumerate(train_x):
        _ts_len = len(item)
        seq_lengths.append(_ts_len)
        _feature_len = len(item[0])
        # _tmp = []
        for idx in range(MAX_LEN - _ts_len):
            item.append([0.0] * _feature_len)
    sorted_seq_len = sorted(list(zip(range(len(seq_lengths)), seq_lengths)), key=lambda x: x[1], reverse=True)
    perm_idx = [item[0] for item in sorted_seq_len]
    sorted_train_x = []
    sorted_train_y = []
    for idx in perm_idx:
        sorted_train_x.append(train_x[idx])
        sorted_train_y.append(train_y[idx])
    del train_x
    del train_y
    tensor_in = torch.FloatTensor(sorted_train_x)
    # tensor_out = torch.FloatTensor(train_y)
    print(tensor_in.size())
    train_x_pack = nn_utils.rnn.pack_padded_sequence(tensor_in, [item[1] for item in sorted_seq_len], batch_first=True)
    train_y_tensor = torch.from_numpy(np.array(sorted_train_y)).float()
    del sorted_train_y
    return train_x_pack, train_y_tensor, seq_lengths


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
    train_x_pack, train_y, seq_lengths = get_data_loader()
    for step in range(3):
        # FIXME generate random test case
        # x_np = np.random.random(BATCH_SIZE * TIME_STEP * INPUT_SIZE).reshape(BATCH_SIZE, TIME_STEP, INPUT_SIZE)
        # y_np = np.random.random(BATCH_SIZE).reshape(BATCH_SIZE, 1)
        #
        # x = torch.from_numpy(x_np).float()
        # y = torch.from_numpy(y_np).float()
        prediction, outputs, h_state = rnn(train_x_pack, h_state)  # rnn对于每一个step的prediction, 还有最后一个step的h_state
        h_state = h_state.data  # 要把h_state 重新包装一下才能放入下一个iteration,不然会报错
        loss = loss_func(outputs, train_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backprogation, compute gradients
        optimizer.step()  # apply gradients
        # if step % 5 == 0:
        plt.cla()
        plt.scatter(range(len(seq_lengths)), train_y[:, 0].data.numpy())
        plt.plot(range(len(seq_lengths)), outputs[:, 0].data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'loss={0},step={1}'.format(loss.data.numpy(), step), fontdict={'size': 20, 'color': 'red'})
        plt.pause(2.5)


if __name__ == '__main__':
    rnn_reg_training()
    # train_x, train_y, seq_length = get_data_loader()
    # print(len(seq_length))
