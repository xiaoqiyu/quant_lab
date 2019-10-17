# -*- coding: utf-8 -*-
# @time      : 2019/10/14 20:08
# @author    : rpyxqi@gmail.com
# @file      : tf_multi_factor.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from quant_models.model_processing.models import Model
from quant_models.data_processing.dataset import DataSet
from quant_models.utils.helper import get_parent_dir
from quant_models.utils.logger import Logger
from tensorflow.python.tools import inspect_checkpoint as chkp

logger = Logger(log_level='DEBUG', handler='ch').get_log()


class TFMultiFactor(Model):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._dataset = None
        self.training_op = None
        self.accuracy = None
        self.sess = None
        self._test_loss = []
        self._model_name = None
        self.summary_op = None

    def build_model(self, feature_hidden_layers=1, all_hidden_layers=2, feature_shape=(), indust_shape=(),
                    save_prefix=''):
        '''
        :param feature_hidden_layers:
        :param all_hidden_layers:
        :param feature_shape: ((sec_num,feature_sub_type_num))
        :param indust_shape: (sec_num, feature_type_num), industry and country
        :param save_prefix:
        :return:
        '''
        assert feature_shape[0][0] == indust_shape[0], (
                "feature.shape: %s industry.shape: %s" % (feature_shape,
                                                          indust_shape))
        _sub_type_num = [item[1] for item in feature_shape]
        _feature_num = sum(_sub_type_num)
        self.feature_inputs = tf.placeholder(tf.float32,
                                             shape=(None, _feature_num),
                                             name='feature_inputs')
        self.indust_inputs = tf.placeholder(tf.float32, shape=(None, indust_shape[1]),
                                            name='industry_inputs')
        self.Y = tf.placeholder(tf.float32, shape=(None, 1))
        prev = 0
        _feature_layer_inputs = []
        for _f_num in _sub_type_num:
            # FIXME _w will be overwrite
            _w = tf.Variable(tf.random_normal([_f_num, 1]))
            _f_val = tf.matmul(self.feature_inputs[:, prev:_f_num], _w)
            _feature_layer_inputs.append(_f_val)
        flattern_indust_input = tf.unstack(self.indust_inputs)
        _feature_layer_inputs.extend(flattern_indust_input)
        feature_val = tf.Variable(_feature_layer_inputs)

        _total_feature_num = len(feature_shape) + indust_shape[1]
        w = tf.Variable(tf.random_normal([_total_feature_num, 1]))
        self.output = tf.matmul(_total_feature_num, w)

        with tf.name_scope('loss'):
            self.loss = tf.losses.mean_squared_error(labels=self.train_Y, predictions=self.output)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(self.loss)

        with tf.name_scope('summary'):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def train_model(self, train_X, train_Y, acc, n_epochs=100, batch_size=50, model_name=None):
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver()
        self._dataset = DataSet(train_X, train_Y, acc)

        with tf.Session() as self.sess:
            train_writer = tf.summary.FileWriter("E:\pycharm\\algo_trading\mi_remote\mi_models\data\models",
                                                 self.sess.graph)
            init.run()
            for epoch in range(n_epochs):
                logger.info('Run the {0} epoch out of {1}, with '.format(epoch, n_epochs))
                for iteration in range(self._dataset.num_examples // batch_size):
                    x_batch, y_batch, acc = self._dataset.next_batch(batch_size)
                    self.sess.run([self.training_op],
                                  feed_dict={self.train_X: x_batch, self.train_Y: y_batch, self.acc_input: acc})
                x_test, y_test, acc_test = self._dataset.next_batch(batch_size)
                test_loss, test_summary = self.sess.run([self.loss, self.summary_op],
                                                        feed_dict={self.train_X: x_test, self.train_Y: y_test,
                                                                   self.acc_input: acc_test})
                self._test_loss.append(test_loss)
                logger.info('epoch: {0}, test_loss:{1}'.format(epoch, test_loss))
                train_writer.add_summary(test_summary, epoch)
            if model_name:
                self._model_name = model_name
                model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
                saver = tf.train.Saver()
                save_path = saver.save(self.sess, model_path)
                logger.info("Saved to path:{0}".format(save_path))
        self._test_loss = np.array(self._test_loss)
        logger.info('mean for test loss:{0}, std:{1}, var:{2}'.format(self._test_loss.mean(), self._test_loss.std(),
                                                                      self._test_loss.var()))

    def save_model(self, model_name):
        self._model_name = model_name
        model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, model_path)
        logger.info("Saved to path:{0}".format(save_path))

    def load_model(self, model_name):
        self._model_name = model_name
        model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess=self.sess,
                      save_path=model_path)

    def predict(self, x=None, acc=None):
        prediction = self.sess.run([self.output, self.tmp_impact, self.perm_impact, self.i_star],
                                   feed_dict={self.train_X: x, self.acc_input: acc})
        return prediction

    def output_model(self, path=None, modle_name=None):
        ret = chkp.print_tensors_in_checkpoint_file(file_name=modle_name, tensor_name=None, all_tensors=True,
                                                    all_tensor_names=True)
        print(ret)


def test_tf_opt():
    _tmp = np.random.random()
    _ = tf.clip_by_value(_tmp, 0, 5)
    # x = tf.Variable()
    x = tf.Variable(_, name='test')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        x1 = sess.run(x)
        print(x1)


if __name__ == '__main__':
    test_tf_opt()
    import numpy as np

    # m = TFRegModel()
    # m.build_model(x_shape=(1, 10), acc_shape=(1, 2))
    # x = np.random.random(1000).reshape(100, 10)
    # y = np.random.random(100).reshape(100, 1)
    # acc = np.random.random(200).reshape(100, 2)
    # m.train_model(x, y, acc, 2, 50, 'tf_dnn')

    # m.save_model('test_dnn')
    # r = m.predict(np.random.random(200).reshape(100, 2), np.random.random(100).reshape(100, 1))
    # print(r)
    # m.load_model('tf_dnn')

    m = TFMultiFactor()
    m.build_model(feature_shape=[(10, 2), (10, 3), (10, 3), (10, 2)], indust_shape=(10, 5))
    x = np.random.random(2000).reshape(200, 10)
    y = np.random.random(200).reshape(200, 1)

    acc = np.random.random(400).reshape(200, 2)
    m.train_model(x, y, acc, 5, 50, 'test')
    # m.output_model('test')
    # m.load_model('test')
    # for i in range(3):
    #     # m.load_model('tf_dnn')
    #     r = m.predict(np.random.random(8).reshape(1, 8), np.random.random(2).reshape(1, 2))
    #     total, tmp, perm, instant = r
    #     print(total[0], tmp[0], perm[0], instant[0])
