#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import numpy as np


class MockRedis(dict):
    """
    Class for mocking Redis store.
    """

    @staticmethod
    def _encode(value):
        """
        Redis stores strings in encoded bytes.

        :param value: value to be encoded
        :return :encoded value (if not `float` or `int)
        """

        if isinstance(value, (int, float, bytes)):
            return value
        else:
            return str(value).encode()

    def rpush(self, queue_name, value):
        """
        Pushes `value` to a queue specified by `queue_name`.

        :param queue_name: name of the queue to be used
        :param value: value to be pushed
        """

        if queue_name not in self:
            self[queue_name] = []

        self[queue_name].append(self._encode(value))

    def set(self, name, value):
        """
        Sets `name` to `value`.

        :param name: key under which the `value` is stored.
        :param value: value to be stored
        """

        self[name] = self._encode(value)

    def delete(self, name):
        """
        Deletes `name` from the store.

        :param name: name to be deleted
        """

        del self[name]

    def get(self, name):
        """
        Retrieves `name` from the store.

        :param name: name to be retrieved
        :return: value as encoded string (to emulate Redis behavior)
        """

        if name in self:
            return self[name]

        return None

    def lrange(self, queue_name, start, end):
        """
        Retrieves values from `queue_name` in a slice
        between the `start` and `end` (incluseive).

        :param queue_name: name of the queue
        :param start: slice start
        :param end: slice end

        :return: list of selected values
        """

        # including the end index
        if queue_name in self:
            return self[queue_name][start:end + 1]

        # redis returns empty list when queue does not exists
        return []

    def ltrim(self, queue_name, start, end):
        """
        Removes all values from `queue_name` not within the slice
        between the `start` and `end` (inclusive).

        :param queue_name: name of the queue
        :param start: slice start
        :param end: slice end
        """

        self[queue_name] = self.lrange(queue_name, start, end)


class MockKerasModel:
    """
    Mocks Keras model.
    """

    def __init__(self, image_shape, n_output):
        """
        Initiates the class.

        :param image_shape: expected shape of input images
        :param n_output: number of units in the output layer
        """

        self.input_shape = (None,) + image_shape
        self._n_output = n_output

    def predict(self, x):
        """
        Mocks the model forward pass.

        :param x: input data
        :return: array containing the output layer result
        """

        if self.input_shape[1:] != x.shape[1:]:
            raise ValueError(f'Input shape {x.shape[1:]} not compatible with '
                             f'{self.input_shape[1:]}.')

        return np.zeros(shape=(x.shape[0], self._n_output), dtype='float32') + 0.5
