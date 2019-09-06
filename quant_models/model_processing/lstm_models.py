# -*- coding: utf-8 -*-
# @time      : 2018/12/3 16:30
# @author    : rpyxqi@gmail.com
# @file      : lstm_models.py

from quant_models.utils.helper import get_config
from quant_models.model_processing.models import Model
import tensorflow as tf
from tensorflow.contrib import rnn
from quant_models.model_processing.dataset import DataSet
from quant_models.utils.logger import Logger
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.init_ops import glorot_uniform_initializer, orthogonal_initializer

import numpy as np
from tensorflow.contrib.layers.python.layers.layers import batch_norm

config = get_config()
logger = Logger(log_level='DEBUG', handler='ch').get_log()


class LSTM_Reg_Model(Model):

    def __init__(self, model_name):
        '''
        Initialize parameters for the LSTM model

        '''
        self.model_name = model_name

    def build_model(self, **kwargs):
        self.step = kwargs.get('step')
        self.input_size = kwargs.get('input_size')
        self.global_step = kwargs.get('global_step')
        self.starter_learning_rate = kwargs.get('starter_learning_rate')
        self.decay_step = kwargs.get('decay_step')
        self.decay_rate = kwargs.get('decay_rate')
        self.hidden_size = kwargs.get('hidden_size')
        self.nclasses = kwargs.get('nclasses')
        self.position = kwargs.get('position')
        self.gamma = None
        self.me_positon = None
        self.summary_op = None
        self.weights = None
        self.biases = None
        self.learning_rate = None
        self.cost = kwargs.get('cost') or 0.0002
        self.starter_learning_rate = kwargs.get('starter_learning_rate')
        self.loss = None
        self.avg_position = None
        self.keep_rate = None
        self.x = None
        self.y = None
        self.is_training = None
        self.build_graph()

    def _create_learning_rate(self):
        '''
        create learning rate
        :return:
        '''
        with tf.variable_scope("parameter"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                            self.decay_step, self.decay_rate, staircase=True,
                                                            name="learning_rate")

    def _create_placeholders(self):
        with tf.variable_scope("input"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.step, self.input_size], name='history_feature')
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='target')
            self.me_positon = tf.placeholder(tf.float32, shape=[None, 1], name='me_position')
            self.pred_return = tf.placeholder(tf.float32, shape=[None, 1], name='target_rate')
            self.is_training = tf.placeholder(tf.bool, name='mode')
            self.keep_rate = tf.placeholder(tf.float32, name='kepp_rate')
            self.gamma = tf.placeholder(tf.float32, name='mi_factor')

    def _create_weights(self):
        with tf.variable_scope("weights"):
            self.weights = {
                'out': tf.get_variable("weights", [self.hidden_size, self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1), )
            }
            self.biases = {
                'out': tf.get_variable("bias", [self.nclasses],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=1))
            }

    def batch_norm_layer(self, signal, scope):
        '''
        batch normalization layer before activation
        :param signal: input signal
        :param scope: name scope
        :return: normalized signal
        '''
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(self.is_training,
                       lambda: batch_norm(signal, is_training=True,
                                          param_initializers={"beta": tf.constant_initializer(3.),
                                                              "gamma": tf.constant_initializer(2.5)},
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=1., scope=scope),
                       lambda: batch_norm(signal, is_training=False,
                                          param_initializers={"beta": tf.constant_initializer(3.),
                                                              "gamma": tf.constant_initializer(2.5)},
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=1.,
                                          scope=scope, reuse=True))

    def _create_loss(self):
        '''
        Risk estimation loss function. The output is the planed position we should hold to next day. The change rate of
        next day is self.y, so we loss two categories of money: - self.y * self.position is trade loss,
        cost * self.position is constant loss because of tax and like missing profit of buying national debt. Therefore,
        the loss function is formulated as: 100 * (- self.y * self.position + cost * self.position) = -100 * ((self.y - cost) * self.position)
        :return:
        '''
        # with tf.device("/cpu:0"):
        xx = tf.unstack(self.x, self.step, 1)
        lstm_cell = rnn.LSTMCell(self.hidden_size, forget_bias=1.0, initializer=orthogonal_initializer())
        dropout_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_rate, output_keep_prob=self.keep_rate,
                                      state_keep_prob=self.keep_rate)

        outputs, states = rnn.static_rnn(dropout_cell, xx, dtype=tf.float32)
        signal = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        scope = "activation_batch_norm"
        norm_signal = self.batch_norm_layer(signal, scope=scope)
        # batch_norm(signal, 0.9, center=True, scale=True, epsilon=0.001, activation_fn=tf.nn.relu6,
        #           is_training=is_training, scope="activation_batch_norm", reuse=False)
        self.position = tf.nn.relu6(norm_signal, name="relu_limit") / 6.
        self.avg_position = tf.reduce_mean(self.position)

        # return prediction
        self.pred_return = tf.nn.relu6(norm_signal) / 2

        market_impact_cost = tf.multiply(self.me_positon, self.position)
        tmp = tf.subtract(self.y, market_impact_cost)
        tmp1 = tf.subtract(tmp, self.cost)

        # trained_loss
        # loss for predict return
        # self.loss = tf.reduce_mean(tf.subtract(self.y, self.pred_return, name='estimated_cost'))
        # loss for predict static position
        # self.loss = -100. * tf.reduce_mean(tf.multiply((self.y - self.cost), self.position, name="estimated_risk"))
        # loss for predict dynamic position
        self.loss = -100. * tf.reduce_mean(tf.multiply(tmp1, self.position, name="estimated_risk"))

    def _create_optimizer(self):
        '''
        create optimizer
        :return:
        '''
        # with tf.device("/cpu:0"):
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, name="optimizer").minimize(self.loss,
                                                                                                  global_step=self.global_step)

    def _create_summary(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("histogram loss", self.loss)
        tf.summary.scalar("average position", self.avg_position)
        tf.summary.histogram("histogram position", self.avg_position)
        self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_learning_rate()
        self._create_placeholders()
        self._create_weights()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()

    # def train_model(self, train_set, val_set, train_steps=10000, batch_size=32, keep_rate=1.):
    def train_model(self, train_X=[], train_Y=[], val_X=[], val_Y=[], **kwargs):
        train_steps = kwargs.get('train_steps') or 10000
        batch_size = kwargs.get('batch_size') or 32
        keep_rate = kwargs.get('keep_rate') or 1
        train_me_pos = kwargs.get('train_me_pos')
        val_me_pos = kwargs.get('val_me_pos')
        gamma = kwargs.get('gamma')
        initial_step = 1
        # val_features = val_set.images
        # val_labels = val_set.labels
        VERBOSE_STEP = 10  # int(len(train_features) / batch_size)
        VALIDATION_STEP = VERBOSE_STEP
        train_set = DataSet(train_X, train_Y, train_me_pos)
        saver = tf.train.Saver()
        min_validation_loss = 100000000.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("../data/models/graphs", sess.graph)
            for i in range(initial_step, initial_step + train_steps):
                batch_features, batch_labels, batch_params = train_set.next_batch(batch_size)
                _, loss, avg_pos, summary = sess.run(
                    [self.optimizer, self.loss, self.avg_position, self.summary_op],
                    feed_dict={self.x: batch_features, self.y: batch_labels,
                               self.is_training: True, self.keep_rate: keep_rate, self.me_positon: batch_params,
                               self.gamma: gamma})
                writer.add_summary(summary, global_step=i)
                if i % VERBOSE_STEP == 0:
                    hint = None
                    if i % VALIDATION_STEP == 0:
                        val_loss, val_avg_pos = sess.run([self.loss, self.avg_position],
                                                         feed_dict={self.x: val_X, self.y: val_Y,
                                                                    self.is_training: False, self.keep_rate: 1.,
                                                                    self.me_positon: val_me_pos, self.gamma: gamma})
                        hint = 'Average Train Loss at step {}: {:.7f} Average position {:.7f}, Validation Loss: {:.7f} Average Position: {:.7f}'.format(
                            i, loss, avg_pos, val_loss, val_avg_pos)
                        if val_loss < min_validation_loss:
                            min_validation_loss = val_loss
                            # TODO change model saving path
                            save_path = saver.save(sess, "../data/models/stock_selection_lstm")
                            logger.info('save path is:', save_path)
                    else:
                        hint = 'Average loss at step {}: {:.7f} Average position {:.7f}'.format(i, loss, avg_pos)
                    logger.debug(hint)

    def predict(self, train_X=[], train_Y=[], me_pos=[], gamma=0.0001, step=30, input_size=61, learning_rate=0.001,
                hidden_size=8, nclasses=1):

        sess = tf.Session()
        # tf.reset_default_graph()
        # model_path = os.path.dirname('../data/models//checkpoint')
        # print('load model from path:{0}'.format(model_path))
        # ckpt = tf.train.get_checkpoint_state(model_path)
        # if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver()
        # saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        saver.restore(sess=sess,
                      save_path='E:\pycharm\\algo_trading\quant_models\quant_models\data\models\stock_selection_lstm')
        # saver.restore(sess=sess, save_path="../data/models/stock_selection_lstm")
        sess.run(tf.global_variables_initializer())
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint/checkpoint'))
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # pred, avg_pos = sess.run([self.position, self.avg_position],
        #                          feed_dict={self.x: train_X, self.y: train_Y,
        #                                     self.is_training: False, self.keep_rate: 1.})
        pred = sess.run([self.position, self.pred_return],
                        feed_dict={self.x: train_X, self.y: train_Y,
                                   self.is_training: False, self.keep_rate: 1.,
                                   self.me_positon: me_pos, self.gamma: gamma})

        return pred
        # print('Predict return')
        # pprint.pprint(pred)

        # cr = calculate_cumulative_return(labels, pred)
        # print("changeRate\tpositionAdvice\tprincipal\tcumulativeReturn")
        # for i in range(len(labels)):
        #     print(str(labels[i]) + "\t" + str(pred[i]) + "\t" + str(cr[i] + 1.) + "\t" + str(cr[i]))

        # print("ChangeRate\tPositionAdvice")
        # for i in range(len(labels)):
        #    print(str(labels[i][0]) + "\t" + str(pred[i][0]))


if __name__ == '__main__':
    # m = LSTM_Reg_Model('lstm')
    # m.build_model(step=5, input_size=10, starter_learning_rate=0.001, learning_rate=0.001, hidden_size=4, nclasses=1,
    #               decay_step=500, decay_rate=1.0, cost=0.0002)
    # m.build_graph()
    train_x = np.random.random(size=(100, 5, 10))
    train_y = np.random.random(size=(100, 1))
    val_x = np.random.random(size=(50, 5, 10))
    val_y = np.random.random(size=(50, 1))
    train_me_pos = np.random.randint(2, 6, size=(100, 1)) / 10.0
    val_me_pos = np.random.randint(2, 6, size=(50, 1)) / 10.0

    # m.train_model(train_X=train_x, train_Y=train_y, val_X=val_x, val_Y=val_y, train_me_pos=train_me_pos,
    #               val_me_pos=val_me_pos, gamma=0.0001, train_steps=10)

    m1 = LSTM_Reg_Model('lstm')
    m1.build_model(step=5, input_size=10, starter_learning_rate=0.001, learning_rate=0.001, hidden_size=4, nclasses=1,
                   decay_step=500, decay_rate=1.0, cost=0.0002)

    m1.train_model(train_X=train_x, train_Y=train_y, val_X=val_x,
                   val_Y=val_y, train_me_pos=train_me_pos,
                   val_me_pos=val_me_pos, gamma=0.0001, train_steps=10)
    pos, ret = m1.predict(val_x, val_y, val_me_pos, 0.0001)
    print(pos)
