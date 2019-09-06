# -*- coding: utf-8 -*-
# @time      : 2018/12/4 10:05
# @author    : rpyxqi@gmail.com
# @file      : dataset.py

import numpy as np


class DataSet(object):
    def __init__(self, images, labels, params):
        # assert len(images) == len(labels)
        # self._num_examples = images.shape[0]
        self._num_examples = len(images)
        # images = images.astype(np.float32)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = self._num_examples
        self._params = params

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            np.random.shuffle(perm)
            np.random.shuffle(perm)
            np.random.shuffle(perm)
            np.random.shuffle(perm)
            # self._images = self._images[perm]
            # self._labels = self._labels[perm]
            perm_image = [self._images[idx] for idx in perm]
            self._images = perm_image
            del perm_image
            perm_label = [self._labels[idx] for idx in perm]
            self._labels = perm_label
            del perm_label
            if isinstance(self._params, np.ndarray) and self._params.shape[0]:
                perm_params = [self._params[idx] for idx in perm]
                self._params = perm_params
                del perm_params

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._params[start:end]
