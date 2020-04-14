#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import cv2
import numpy as np
import pandas as pd
from .constants import (DEFAULT_IMAGE_SHAPE, IMAGE_ID_COL, RLE_MASK_COL,
                        IMAGE_LABEL_COL)


def decode_rle(rle, shape=DEFAULT_IMAGE_SHAPE):
    """
    Decodes RLE into a binary image mask.

    :param rle: rle string containing the mask
    :shape: tuple defining the image width and height

    :return: decoded mask as numpy array
    """

    mask_1d = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    rle = rle.split()
    positions = zip(rle[0::2], rle[1::2])
    positions = map(lambda x: (int(x[0]) - 1, int(x[1])), positions)

    for position in positions:
        start = position[0]
        end = position[0] + position[1]
        mask_1d[start:end] = 1

    return mask_1d.reshape(shape).T


def resize_image(image, scale):
    """
    Resises an image using the given scale.

    :param image: numpy array containing the image
    :param scale: resizing ratio
    :return: numpy array with the resized image
    """

    scale = abs(scale)

    # handle rgb images with shape (n, m, 3)
    dims = image.shape if len(image.shape) == 2 else image.shape[:2]

    width, height = tuple(int(dim * scale) for dim in dims)

    return cv2.resize(image, (width, height))


def get_image_labels(ground_truth):
    """
    Converts ground truth dataframe to global binary image labels.

    :param ground_truth: dataframe containing the image ground truth with RLE masks
    :return: dataframe containing image labels
    """

    image_labels = ground_truth.groupby([IMAGE_ID_COL]).agg(
        {RLE_MASK_COL: lambda x: 0 if pd.isnull(x).all() else 1}).reset_index()

    return image_labels.rename(columns={RLE_MASK_COL: IMAGE_LABEL_COL})
