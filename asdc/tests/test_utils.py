#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import cv2
import base64
import numpy as np
import pandas as pd
from unittest import TestCase, main
from ..core.constants import IMAGE_ID_COL, RLE_MASK_COL, DEFAULT_IMAGE_SIZE
from ..core.utils import (decode_rle, rescale, check_square_size,
                          get_image_rle_masks, decode_image_b64,
                          check_image_rgb, convert_history,
                          ImageMaskDownsampler)


class ConvertHistoryTest(TestCase):
    """
    Tests `utils.convert_history` function.
    """

    def test_convert_history(self):
        """
        Tests that the function produces an
        identical history dictionary with the
        correct datatypes.
        """

        test_case_history = {
            'loss': [np.float64(0.1), np.float64(0.02)],
            'val_loss': [np.float64(0.11), np.float64(0.019)]}

        expected = {'loss': [0.1, 0.02], 'val_loss': [0.11, 0.019]}

        self.assertDictEqual(
            convert_history(test_case_history),
            expected)


class CheckImageRgbTest(TestCase):
    """
    Tests `utils.check_image_rgb` function.
    """

    def test_is_rgb(self):
        """
        Tests that exception is not raised.
        """

        test_case_image = np.zeros(shape=(10, 10, 3))
        check_image_rgb(test_case_image)

    def test_not_rgb(self):
        """
        Tests that exception is raised.
        """

        for test_case_shape in ((10, 10), (20, 20, 1)):
            with self.assertRaises(ValueError):
                check_square_size(np.zeros(shape=test_case_shape))


class CheckSquareSizeTest(TestCase):
    """
    Tests `utils.check_square_size` function.
    """

    def test_is_square(self):
        """
        Tests that exception is not raised.
        """

        for test_case_size in ((1, 1), (20, 20), (40, 40)):
            check_square_size(test_case_size)

    def test_not_square(self):
        """
        Tests that exception is raised.
        """

        for test_case_size in ((1, 2), (20, 24)):
            with self.assertRaises(ValueError):
                check_square_size(test_case_size)


class DecodeRleTest(TestCase):
    """
    Tests `utils.decode_rle` function.
    """

    def test_decode_rle(self):
        """
        Tests correct decoding.
        """

        test_case_rle = '11 5 20 5'
        test_case_size = (6, 6)
        expected = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0]])

        np.testing.assert_array_equal(
            decode_rle(test_case_rle, test_case_size), expected)


class RescaleTest(TestCase):
    """
    Tests `utils.rescale` function.
    """

    def test_rescale(self):
        """
        Test correct rescaling.
        """

        test_case_image = (np.eye(10, 10) * 255).astype('uint8')
        test_case_scale = 0.2
        expected = np.array([[255, 0], [0, 255]])

        np.testing.assert_array_equal(
            rescale(test_case_image, test_case_scale), expected)


class GetImageRleMasksTest(TestCase):
    """
    Tests `utils.get_image_rle_masks` function.
    """

    def test_get_image_rle_masks(self):
        """
        Tests encoded masks are correctly computed.
        """

        test_case_ground_truth = pd.DataFrame({
            IMAGE_ID_COL: ['test_1', 'test_2', 'test_2', 'test_3'],
            RLE_MASK_COL: [np.nan, '10 17', '59 2', '0']})

        expected = pd.DataFrame({
            IMAGE_ID_COL: ['test_1', 'test_2', 'test_3'],
            RLE_MASK_COL: [np.nan, '10 17 59 2', '0']})

        pd.testing.assert_frame_equal(
            get_image_rle_masks(test_case_ground_truth),
            expected)


class ImageMaskDownsamplerTest(TestCase):
    """
    Tests `utils.ImageDownSampler`.
    """

    def test_size_not_correct(self):
        """
        Test that exception is raise when mask sizes not correct.
        """

        for test_case_output_size in ((128, 129), (129, 129)):
            with self.assertRaises(ValueError):
                _ = ImageMaskDownsampler(output_size=test_case_output_size)

    def test_downsample(self):
        """
        Tests `utils.ImageDownsampler.downsample` correctly
        downsamples masks.
        """

        downsampler = ImageMaskDownsampler(output_size=(128, 128))

        test_case_mask = np.zeros(shape=DEFAULT_IMAGE_SIZE)
        test_case_mask[0, 0] = 1
        test_case_mask[-1, -1] = 1

        for test_case_output_size in ((128, 128), (2, 2), (1, 1)):

            downsampler = ImageMaskDownsampler(output_size=test_case_output_size)
            result = downsampler.downsample(test_case_mask)

            expected = np.zeros(shape=test_case_output_size)
            expected[0, 0] = 1
            expected[-1, -1] = 1

            np.testing.assert_array_equal(result, expected)


class DecodeImageB64Test(TestCase):
    """
    Tests `utils.decode_image_b64` function.
    """

    def test_decode_image_b64(self):
        """
        Tests that the image is correctly decoded.
        """

        test_case_image = np.zeros(shape=(50, 50, 3))
        test_case_image_bytes = cv2.imencode('.png', test_case_image)[1].tobytes()
        test_case_image_b64 = base64.b64encode(test_case_image_bytes).decode()

        result = decode_image_b64(test_case_image_b64)
        np.testing.assert_array_equal(result, test_case_image)


if __name__ == '__main__':
    main()
