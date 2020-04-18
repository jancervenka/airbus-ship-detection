#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import json
from unittest import TestCase, mock, main
from .utils import MockRedis
from ..service.detect import process_request, store_get, store_push


class StorePushTest(TestCase):
    """
    Tests `detect.store_push` function.
    """

    def test_store_push(self):
        """
        Tests that data can be pushed.
        """

        db = MockRedis()
        test_case_image_b64 = 'test'
        store_push(db, test_case_image_b64)
        self.assertTrue(len(db), 1)


class StoreGetTest(TestCase):
    """
    Tests `detect.store_get` function.
    """

    @mock.patch('time.sleep', lambda _: None)
    @mock.patch('asdc.service.detect.REQUEST_TIMEOUT_SEC', 0.5)
    def test_store_get_timeout(self):
        """
        Tests the timeout when no result found.
        """

        db = MockRedis()
        result = store_get(db, 'test')
        expected = {'error': 'request_timeout'}

        self.assertDictEqual(result, expected)

    def test_store_get(self):
        """
        Tests that the result can be successfully
        retrieved from the store.
        """

        test_case_result = {'test': 'test'}
        test_case_request_id = 1

        db = MockRedis()
        db.set(test_case_request_id, json.dumps(test_case_result))

        result = store_get(db, test_case_request_id)
        expected = test_case_result

        self.assertDictEqual(result, expected)
        self.assertEqual(len(db), 0)


class ProcessRequestTest(TestCase):
    """
    Tests `detect.process_request_function`.
    """

    def test_no_data(self):
        """
        Tests that correct error is return when not json data
        in the payload.
        """

        expected = {'error': 'no_json_data'}
        result = process_request(MockRedis(), {})

        self.assertDictEqual(result, expected)


if __name__ == '__main__':
    main()
