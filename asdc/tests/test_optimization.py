#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import numpy as np
import pandas as pd
from unittest import TestCase, main
from .utils import MockKerasModel
from ..core.constants import IMAGE_ID_COL, RLE_MASK_COL
from ..core.optimization import RandomSearch


class RandomSearchTest(TestCase):
    """
    Tests `optimization.RandomSearch` class.
    """

    @staticmethod
    def _mock_model_factory(*args, **kwargs):
        """
        Function generating mock model instances.

        :return: `utils.MockKerasModel` instance
        """

        return MockKerasModel((1, 1), 1)

    def setUp(self):
        """
        Sets up the tests.
        """

        test_param_grid = {'lr': [1, 2], 'n_dense': [1, 2]}

        self._random_search = RandomSearch(
            model_factory=self._mock_model_factory,
            image_directory=None,
            image_shape=(5, 5, 3),
            cv=5,
            param_grid=test_param_grid)

        self._test_image_rle_masks = pd.DataFrame({
            IMAGE_ID_COL: [str(i) for i in range(128)],
            RLE_MASK_COL: [np.nan, '1 10'] * 64})

    def test_init_generator(self):
        """
        Tests that `optimization.RandomSearch._init_generator`
        corretly creates a generator from the train/validation
        subset.
        """

        test_case_ix = np.arange(0, 128, 2)

        gen = self._random_search._init_generator(
            self._test_image_rle_masks, test_case_ix)

        expected_image_ids = self._test_image_rle_masks.loc[
            test_case_ix,
            IMAGE_ID_COL].tolist()

        self.assertEqual(len(gen), 64 // gen._batch_size)
        self.assertListEqual(gen._image_ids, expected_image_ids)

    def test_eval_params(self):
        """
        Tests that `optimization.RandomSearch._eval_params` correctly
        passes trough all the cross-validation folds.
        """

        result = self._random_search._eval_params(
            self._test_image_rle_masks, {'test': None})

        expected = {'mean_loss': 0, 'params': {'test': None}, 'losses': [0] * 5}
        self.assertDictEqual(result, expected)


if __name__ == '__main__':
    main()
