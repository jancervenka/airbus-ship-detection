#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import json
import numpy as np
from unittest import TestCase, main
from .utils import MockRedis, MockKerasModel
from ..service.constants import REQUEST_BATCH_SIZE, QUEUE_NAME
from ..service.backend import RequestProcessor


class BackendProcessorTest(TestCase):
    """
    Tests `backend.RequestProcessor` class.
    """

    def setUp(self):
        """
        Sets up the tests.
        """

        self._test_image_shape = (10, 10, 3)
        self._test_n_output = 4

        # png image ones(5, 5, 3)
        self._test_image_b64 = ''.join(
            ('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAE',
             '0lEQVQIHWMEAgYGBkYgYGBgAAAAVgAJ0q8TBQAAAABJRU5ErkJggg=='))

        # decoded and resized self._test_image_b64
        self._test_image = np.ones(
            shape=self._test_image_shape).astype('float32') / 255

        self._db = MockRedis()
        self._model = MockKerasModel(
            image_shape=self._test_image_shape, n_output=self._test_n_output)

        self._request_processor = RequestProcessor(
            db=self._db, model=self._model)

    def test_get_requests(self):
        """
        Tests correct number of values is retrieved from the
        Redis queue.
        """

        self.assertListEqual(self._request_processor._get_requests(), [])

        def test_case_wrapper(n_to_push, n_expected):
            """
            Test case wrapper.

            :param n_to_push: number of values to be pushed
                              to the queue
            :param n_expected: number of values expected
                               to be retrieved from the queue
            """

            for _ in range(n_to_push):
                self._db.rpush(QUEUE_NAME, {})

            self.assertListEqual(
                self._request_processor._get_requests(), [{}] * n_expected)

        test_case_wrapper(3, 3)
        test_case_wrapper(REQUEST_BATCH_SIZE + 10, REQUEST_BATCH_SIZE)

        self._db.delete(QUEUE_NAME)

    def test_resize_if_required(self):
        """
        Tests that `backend.RequestProcessor._resize_if_required`
        correctly resizes an image if necessary.
        """

        expected = np.zeros(shape=self._test_image_shape)

        tests = (
            (expected, expected),
            (np.zeros(shape=(28, 28, 3)), expected),
            (np.zeros(shape=(5, 5, 3)), expected))

        for test_case_image, expected in tests:
            np.testing.assert_array_equal(
                self._request_processor._resize_if_required(test_case_image),
                expected)

    def test_create_image_failure(self):
        """
        Tests that `backend.RequestProcessor._create_image` returns
        `None` if an image cannot be created.
        """

        self.assertIsNone(self._request_processor._create_image('test'))

    def test_create_image_success(self):
        """
        Tests that `backend.RequestProcessor._create_image` can
        successfully create an image.
        """

        expected = self._test_image
        np.testing.assert_array_equal(
            self._request_processor._create_image(self._test_image_b64),
            expected)

    def test_prepare_ok_nok_requests_empty(self):
        """
        Tests that `backend.RequestProcessor._prepare_ok_nok_requests`
        can handle empty requests.
        """

        for result in self._request_processor._prepare_ok_nok_requests([]):
            self.assertListEqual(list(result), [])

    def test_prepare_ok_nok_requests(self):
        """
        Test that `backend.RequestProcessor._prepare_ok_nok_requests`
        correctly prepares and filters ok and nok requests.
        """

        test_case_requests = [
            {'id': 1, 'image_b64': self._test_image_b64},
            {'id': 2, 'image_b64': 'test'},
            {'id': 3, 'image_b64': 'test'},
            {'id': 4, 'image_b64': self._test_image_b64}]

        expected_ok = [(self._test_image, 1), (self._test_image, 4)]
        expected_nok = [(None, 2), (None, 3)]

        result_ok, result_nok = self._request_processor._prepare_ok_nok_requests(
            test_case_requests)

        def assert_prepared_requests_equal(result, expected):
            """
            Asserts that a list produced by the function
            is equal to the expected list.
            """

            for r_r, r_e in zip(result, expected):

                # compare id
                self.assertEqual(r_r[1], r_e[1])
                # compare arrays
                if isinstance(r_e[0], np.ndarray):
                    np.testing.assert_array_equal(r_r[0], r_e[0])
                else:
                    self.assertIsNone(r_e[0])
                    self.assertIsNone(r_e[0])

        assert_prepared_requests_equal(list(result_ok), expected_ok)
        assert_prepared_requests_equal(list(result_nok), expected_nok)

    def test_process_nok_requests(self):
        """
        Tests that `backend.RequestProcessor._process_nok_requests`
        correctly process nok requests and stores them in the Redis.
        """

        test_case_nok_requests = ((None, 't_1'), (None, 't_2'))
        self._request_processor._process_nok_requests(test_case_nok_requests)

        for key in ('t_1', 't_2'):
            self.assertDictEqual(json.loads(self._db[key]), {'error': 'image_not_compatible'})

        self._db.delete('t_1')
        self._db.delete('t_2')

    def test_process_ok_requests(self):
        """
        Tests that `backend.RequestProcessor._process_ok_requests`
        correctly process nok requests and stores them in the Redis.
        """

        # tests empty requests, nothing to process, empty Redis
        self._request_processor._process_ok_requests(tuple())
        self.assertEqual(len(self._db), 0)

        # tests one requests
        test_case_ok_requests = ((self._test_image, 't_3'),)
        self._request_processor._process_ok_requests(test_case_ok_requests)
        self.assertDictEqual(json.loads(self._db['t_3']), {'prediction': 0.5})

        # tests n requests
        test_case_ok_requests = ((self._test_image, 't_4'),
                                 (self._test_image, 't_5'))
        self._request_processor._process_ok_requests(test_case_ok_requests)

        # test everything is stored, then cleaup
        for request_id in ('t_3', 't_4', 't_5'):
            self.assertDictEqual(json.loads(self._db[request_id]), {'prediction': 0.5})
            self._db.delete(request_id)


if __name__ == '__main__':
    main()
