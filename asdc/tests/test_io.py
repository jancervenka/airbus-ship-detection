#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import numpy as np
from tensorflow.keras.utils import to_categorical
from ..core.io import load_image, ImageBatchGenerator
from unittest import TestCase, mock, main


def _mock_imread(image_file_path):
    """
    Mocks `cv2.imread`.

    :param image_file_path: mock path to the image,
                            image name must be a numeric value
    :return: mock image, pixel values equal to the
             numeric image name
    """

    image = np.zeros(shape=(5, 5, 3))
    return image + float(image_file_path.split('/')[-1])


class LoadImageTest(TestCase):
    """
    Tests `io.load_image`.
    """

    @mock.patch('cv2.imread', lambda _: None)
    def test_image_not_found(self):
        """
        Tests that exception is raised when image is not found.
        """

        with self.assertRaises(FileNotFoundError):
            _ = load_image('test')

    @mock.patch('cv2.imread', lambda _: np.zeros(shape=(20, 20)))
    def test_load_resize_expand(self):
        """
        Test that image is successfully loaded, rescaled,
        and dimensions are correctly expanded.
        """

        result = load_image('test', scale=0.5)
        expected = np.zeros(shape=(10, 10, 1))
        np.testing.assert_array_equal(result, expected)


class ImageBatchGeneratorTest(TestCase):
    """
    Tests `io.ImageBatchGenerator` class. Tests that
    batches are correctly defined and that images with
    labels can be retrieved.
    """

    def setUp(self):
        """
        Sets up the tests.
        """

        image_ids = map(str, range(1, 7))
        self._image_labels = {k: v for k, v in zip(image_ids, [0, 1] * 3)}

        self._gen = ImageBatchGenerator(
            image_directory='test',
            image_labels=self._image_labels,
            batch_size=4,
            shuffle=False,
            image_scale=None)

    def test_len(self):
        """
        Tests that generator length (number of batches)
        is correct.
        """

        self.assertTrue(len(self._gen), 2)

    def test_index_out_of_bounds(self):
        """
        Tests `IndexError` is raised when
        accessing out of bounds index.
        """

        with self.assertRaises(IndexError):
            self._gen[2]

    def test_get_image_file_paths(self):
        """
        Tests that `io.ImageBatchGenerator._get_image_file_paths`
        correctly assembles file paths from image ids.
        """

        result = self._gen._get_image_file_paths(['1', '2'])
        expected = ['test/1', 'test/2']
        self.assertListEqual(result, expected)

    @mock.patch('cv2.imread', _mock_imread)
    def test_get_mean(self):
        """
        Tests that `io.ImageBatchGenerator._get_mean`
        correctly computes mean pixel value.
        """

        result = self._gen._get_mean()
        expected = np.mean([i / 255 for i in range(1, 7)])
        self.assertAlmostEqual(result, expected, places=6)

    @mock.patch('cv2.imread', _mock_imread)
    def test_getitem(self):
        """
        Tests that correct batches are successfully
        retrieved.
        """

        def get_expected_x_y(n, id_offset):
            """
            Creates expected results.

            :param n: number of expected images
            :param id_offset: start of the id range
                              of the expected images
            :return: expected x and y arrays 
            """

            expected_x = np.zeros(shape=(n, 5, 5, 3))
            for i in range(n):
                expected_x[i, :, :, :] = (id_offset + i + 1) / 255

            expected_y = to_categorical(np.array([0, 1] * (n // 2)))
            return expected_x, expected_y

        for i, n, offset in ((0, 4, 0), (1, 2, 4)):
            result_x, result_y = self._gen[i]
            expected_x, expected_y = get_expected_x_y(n, offset)

            np.testing.assert_array_equal(result_y, expected_y)
            np.testing.assert_array_almost_equal(result_x, expected_x, decimal=6)


class ImageBatchGeneratorShuffleTest(TestCase):
    """
    Tests image shuffeling in `io.ImageBatchGenerator`.
    """

    def test_shuffle(self):
        """
        Tests image id indexes are in different
        order after epoch end.
        """

        np.random.seed(0)

        gen = ImageBatchGenerator(
            image_directory=None,
            image_labels={'1': 0, '2': 1, '3': 0})

        indexes_before = gen._indexes
        gen.on_epoch_end()
        indexes_after = gen._indexes

        expected_before = np.array([2, 1, 0])
        expected_after = np.array([2, 0, 1])

        np.testing.assert_array_equal(indexes_before, expected_before)
        np.testing.assert_array_equal(indexes_after, expected_after)


if __name__ == '__main__':
    main()
