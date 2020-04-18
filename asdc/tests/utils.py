#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka


class MockRedis(dict):
    """
    Class for mocking Redis store.
    """

    def rpush(self, queue_name, value):
        """
        Pushes `value` to a queue specified by `queue_name`.

        :param queue_name: name of the queue to be used
        :param value: value to be pushed
        """

        if queue_name not in self:
            self[queue_name] = []

        self[queue_name].append(value)

    def set(self, name, value):
        """
        Sets `name` to `value`.

        :param name: key under which the `value` is stored.
        :param value: value to be stored
        """

        self[name] = value

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
            return str(self[name]).encode()

        return None
