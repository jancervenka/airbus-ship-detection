#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import cv2
import base64
import itertools
import numpy as np
import pandas as pd
from .constants import (DEFAULT_IMAGE_SIZE, IMAGE_ID_COL, RLE_MASK_COL)


def check_square_size(size):
    """
    Raises an exception if `size` is not a square.

    :param size: image/mask size to check
    """

    if size[0] != size[1]:
        raise ValueError('Masks/images must be square-sized.')


def decode_rle(rle, size=DEFAULT_IMAGE_SIZE):
    """
    Decodes RLE into a binary image mask.

    :param rle: rle string containing the mask
    :shape: tuple defining the image width and height

    :return: decoded mask as numpy array
    """

    mask_1d = np.zeros(size[0] * size[1], dtype=np.uint8)

    rle = rle.split()
    positions = zip(rle[0::2], rle[1::2])
    positions = map(lambda x: (int(x[0]) - 1, int(x[1])), positions)

    for position in positions:
        start = position[0]
        end = position[0] + position[1]
        mask_1d[start:end] = 1

    return mask_1d.reshape(size).T


def rescale(image, scale):
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


def get_image_rle_masks(ground_truth):
    """
    Converts ground truth dataframe to RL encoded image masks (each mask contains
    all the target objects in the image)

    :param ground_truth: dataframe containing the image ground truth with RLE masks
    :return: dataframe containing image labels
    """

    image_rle_masks = ground_truth.groupby([IMAGE_ID_COL]).agg({
        RLE_MASK_COL: lambda x: np.nan if pd.isnull(x).all() else ' '.join(x)})

    return image_rle_masks.reset_index()


def decode_image_b64(image_b64):
    """
    Creates numpy array from base64 string containing an image.

    :param image_b64: image base64 string
    :return: image as numpy array
    """

    decoded_bytes = base64.b64decode(image_b64)
    return cv2.imdecode(np.frombuffer(decoded_bytes, np.uint8), cv2.IMREAD_UNCHANGED)


class ImageMaskDownsampler:
    """
    Downsample image target mask using disjunct max pooling.
    """
    # TODO: test this works with output_size=(1, 1) same as the old
    # get_image_labels

    def __init__(self, output_size, input_size=DEFAULT_IMAGE_SIZE):
        """
        Initiates the class.

        :param output_size: `(width, height)` of the masks on the output
        :param input_size: `(width, height)` of the masks on the input
        """

        self._check_size(input_size, output_size)

        self._step = input_size[0] // output_size[0]
        self._all_ij = list(itertools.product(
            range(0, input_size[0], self._step),
            range(0, input_size[1], self._step)))

    @staticmethod
    def _check_size(input_size, output_size):
        """
        Raises an exception if `input_size` is not divisible
        by `output_size` without a remainder or if `input_size`
        or `output_size` are not rectangular.

        :param input_size: `(width, height)` of the masks on the input
        :param output_size: `(width, height)` of the masks on the output
        """

        for size in (input_size, output_size):
            check_square_size(size)

        for i in (0, 1):
            if not input_size[i] % output_size[i] == 0:
                raise ValueError('input_size must be divisble by '
                                 'output_size with no remainder.')

    def downsample(self, mask):
        """
        Downsample a mask.

        :param mask: mask to be downsample
        :return: downsampled mask
        """

        new_mask = np.zeros(shape=tuple(x // self._step for x in mask.shape))

        for i, j in self._all_ij:
            new_mask[i // self._step,
                     j // self._step] = np.max(mask[i:i + self._step, j:j + self._step])

        return new_mask
