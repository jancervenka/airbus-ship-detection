#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import numpy as np
import pandas as pd
from unittest import TestCase, main
from ..core.utils import decode_rle, resize_image, get_image_labels


class DecodeRleTest(TestCase):
    """
    Tests `utils.decode_rle` function.
    """

    def test_decode_rle(self):
        """
        Tests correct decoding.
        """

        test_case_rle = '11 5 20 5'
        test_case_shape = (6, 6)
        expected = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0]])

        np.testing.assert_array_equal(
            decode_rle(test_case_rle, test_case_shape), expected)


class ResizeImageTest(TestCase):
    """
    Tests `utils.resize_image` function.
    """

    def test_resize_image(self):
        """
        Test correct resizing.
        """

        test_case_image = (np.eye(10, 10) * 255).astype('uint8')
        test_case_scale = 0.2
        expected = np.array([[255, 0], [0, 255]])

        np.testing.assert_array_equal(
            resize_image(test_case_image, test_case_scale), expected)


class GetImageLabelsTest(TestCase):
    """
    Tests `utils.get_image_labels` function.
    """

    def test_get_image_labels(self):
        """
        Tests image labels are correctly computed.
        """

        test_case_ground_truth = pd.DataFrame(
            {'ImageId': ['test_1', 'test_2', 'test_2', 'test_3'],
             'EncodedPixels': [np.nan, '0', '0', '0']})

        expected = pd.DataFrame({
            'ImageId': ['test_1', 'test_2', 'test_3'],
            'Label': [0, 1, 1]})

        pd.testing.assert_frame_equal(
            get_image_labels(test_case_ground_truth),
            expected)


if __name__ == '__main__':
    main()
