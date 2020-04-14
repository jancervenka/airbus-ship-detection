#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import pandas as pd
from unittest import TestCase, main
from ..core.training import BalancedImageLabels
from ..core.constants import IMAGE_ID_COL, IMAGE_LABEL_COL


class BalancedImageLabelsTest(TestCase):
    """
    Tests `training.BalancedImageLabels`.
    """

    def setUp(self):
        """
        Sets up the test.
        """

        self._test_image_labels = pd.DataFrame({
            IMAGE_ID_COL: list(range(100)),
            IMAGE_LABEL_COL: [0] * 60 + [1] * 40})

    def test_balance(self):
        """
        Tests that `training.BalancedImageLabels._balance`
        creates correctly balanced image labels.
        """

        result = BalancedImageLabels._balance(self._test_image_labels, 30)
        self.assertTrue(len(result), 30)
        for label in (0, 1):
            self.assertTrue((result[IMAGE_LABEL_COL] == label).sum(), 15)

    def test_to_dict(self):
        """
        Tests that `training.BalancedImageLabels._to_dict`
        correctly converts image labels from dataframe to dictionary
        """

        result = BalancedImageLabels._to_dict(self._test_image_labels)
        expected = {k: v for k, v in zip(range(100), [0] * 60 + [1] * 40)}
        self.assertDictEqual(result, expected)

    def test_split(self):
        """
        Tests that `training.BalancedImageLabels._split` creates
        correct training, validation, test splits.
        """

        result_train, result_test, result_val = BalancedImageLabels._split(
            self._test_image_labels)

        tests = ((result_train, 56), (result_val, 19), (result_test, 25))
        for result, len_ in tests:
            self.assertTrue(len(result), len_)


if __name__ == '__main__':
    main()
