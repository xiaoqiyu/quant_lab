# -*- coding: utf-8 -*-
# @time      : 2019/12/11 18:45
# @author    : rpyxqi@gmail.com
# @file      : torch_reg_models.py


import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)
# HYPER Parameters
TIME_STEP = 10  # rnn time step/image height
INPUT_SIZE = 2
LR = 0.2
DOENLOAD_MINST = False
BATCH_SIZE = 10
HIDDEN_SIZE = 32
NUM_LAYER = 2


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
    for step in range(10):
        # FIXME generate random test case
        x_np = np.random.random(BATCH_SIZE * TIME_STEP * INPUT_SIZE).reshape(BATCH_SIZE, TIME_STEP, INPUT_SIZE)
        y_np = np.random.random(BATCH_SIZE).reshape(BATCH_SIZE, 1)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        prediction, outputs, h_state = rnn(x, h_state)  # rnn对于每一个step的prediction, 还有最后一个step的h_state
        h_state = h_state.data  # 要把h_state 重新包装一下才能放入下一个iteration,不然会报错
        loss = loss_func(outputs, y)
        # loss = loss_func(prediction, y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backprogation, compute gradients
        optimizer.step()  # apply gradients
        # if step % 5 == 0:
        #     plt.cla()
        #     plt.scatter(range(x.shape[0]), y[:,0].data.numpy())
        #     plt.plot(range(x.shape[0]), outputs[:, 0].data.numpy(), 'r-', lw=5)
        #     plt.text(0.5, 0, 'loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        #     plt.pause(.5)


if __name__ == '__main__':
    rnn_reg_training()
