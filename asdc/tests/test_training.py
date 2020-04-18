#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import numpy as np
import pandas as pd
from unittest import TestCase, main
from ..core.training import BalancedImageRleMasks
from ..core.constants import IMAGE_ID_COL, RLE_MASK_COL


class BalancedImageRleMasksTest(TestCase):
    """
    Tests `training.BalancedImageRleMasks`.
    """

    def setUp(self):
        """
        Sets up the test.
        """

        self._test_image_rle_masks = pd.DataFrame({
            IMAGE_ID_COL: list(range(100)),
            RLE_MASK_COL: [np.nan] * 60 + ['2 40'] * 40})

    def test_balance(self):
        """
        Tests that `training.BalancedImageRleMasks._balance`
        creates correctly balanced image masks.
        """

        result = BalancedImageRleMasks._balance(
            self._test_image_rle_masks, sample_size=30, pos_share=0.33)
        self.assertTrue(len(result), 30)
        self.assertEqual((~result[RLE_MASK_COL].isna()).sum(), 10)

    def test_to_dict(self):
        """
        Tests that `training.BalancedImageRleMasks._to_dict`
        correctly converts image masks from dataframe to dictionary
        """

        result = BalancedImageRleMasks._to_dict(self._test_image_rle_masks)
        expected = {k: v for k, v in zip(range(100), [np.nan] * 60 + ['2 40'] * 40)}
        self.assertDictEqual(result, expected)

    def test_split(self):
        """
        Tests that `training.BalancedImageRleMasks._split` creates
        correct training, validation, test splits.
        """

        result_train, result_test, result_val = BalancedImageRleMasks._split(
            self._test_image_rle_masks)

        tests = ((result_train, 56), (result_val, 19), (result_test, 25))
        for result, len_ in tests:
            self.assertTrue(len(result), len_)


if __name__ == '__main__':
    main()
