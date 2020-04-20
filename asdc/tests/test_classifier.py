#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import numpy as np
from unittest import TestCase, main, skip
from ..core.classifier import MaskDetection


class MaskDetectionTest(TestCase):
    """
    Tests `classifier.MaskDetection` class.
    """

    @skip('Not required')
    def test_create_model(self):
        """
        Tests that `classifier.MaskDetection.create_model`
        produces a model with corect input and output shapes.
        """

        test_image_shape = (20, 20, 3)

        model = MaskDetection.create_model(
            image_shape=test_image_shape,
            n_output=4)

        np.testing.assert_array_equal(model.input_shape, (None,) + test_image_shape)

        result = model.predict(np.array([np.ones(shape=test_image_shape)]))
        self.assertTupleEqual(result.shape, (1, 4))


if __name__ == '__main__':
    main()
