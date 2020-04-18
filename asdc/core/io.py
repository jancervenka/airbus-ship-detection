#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import Sequence
from .utils import decode_rle, check_square_size, ImageMaskDownsampler


def load_image(image_file_path, size=None):
    """
    Loads an image.

    :param image_file_path: path to the image
    :param size: tuple `(width, height)`, containing
                 the target size
    :return: image as numpy array
    """

    image = cv2.imread(image_file_path)
    if image is None:
        raise FileNotFoundError(f'Image {image_file_path} not found.')

    if size is not None:
        image = cv2.resize(image, size)

    # change volume (n, m) to (n, m, 1)
    return np.expand_dims(image, axis=2) if len(image.shape) == 2 else image


class ImageBatchGenerator(Sequence):
    """
    Batch image generator. Iterable where each item
    contains one batch of images and target masks (as a tuple
    of two numpy arrays). Each pass through the generator
    (triggered by `self.on_epoch_end`) reshuffles the order
    of images in the batches.

    Batches are loaded on demand (lazy iteration).

    This generator supports parallel processing.
    """
    # TODO: use mean image
    # TODO: handle special case for downsampling mask (1, 1) for speed

    def __init__(self, image_directory, image_rle_masks, mask_size,
                 image_size=None, batch_size=32, shuffle=True):
        """
        Initiates the class.

        :param image_directory: path to the image directory
        :param encoded_masks: dictionary `{image_id: rle,}` containing
                              all images and their encoded masks that
                              should be included in the generator.
        param mask_size: tuple `(width, height)` defining the resolution
                         of the model target masks
        :param image_size: if not `None`, loaded images will be resized;
                           expecting tuple `(width, height)`
        :param batch_size: size of one image batch
        :param shuffle: if `True`, reshufles image id indexes
                        making different image/batch order for each epoch
        """

        for size in (mask_size, image_size):
            if size is not None:
                check_square_size(size)

        self._image_directory = image_directory
        self._image_rle_masks = image_rle_masks
        self._image_ids = list(image_rle_masks.keys())
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._image_size = image_size
        self._mask_size = mask_size
        self._mask_downsampler = ImageMaskDownsampler(output_size=mask_size)
        self.on_epoch_end()

    def __len__(self):
        """
        Computes generator length in number of batches.
        (one epoch = one pass through the generator)

        :return: length
        """

        return int(np.ceil(len(self._image_rle_masks) / self._batch_size))

    def _check_index(self, index):
        """
        Raises an exception if the `index` is out of bounds.

        :param index: index for `self.__getitem__`
        """

        if index >= len(self):
            raise IndexError(f'Generator contains only {len(self)} batches.')

    def __getitem__(self, index):
        """
        Generates data for given batch specified
        by the (batch) `index`.

        :param index: batch index
        :return: tuple of numpy arrays (x, y)
        """

        self._check_index(index)

        start = index * self._batch_size
        end = (index + 1) * self._batch_size
        indexes_in_batch = self._indexes[start:end]

        selected_image_ids = [self._image_ids[i] for i in indexes_in_batch]
        return self._generate_data(selected_image_ids)

    def on_epoch_end(self):
        """
        Reshuffles `_indexes` used to retrieve image ids for
        a batch. During each epoch, the batches contain images in
        a unique order.
        """

        self._indexes = np.arange(len(self._image_ids))
        if self._shuffle:
            np.random.shuffle(self._indexes)

    def _get_image_file_paths(self, image_ids):
        """
        Creates a list of image file paths from a list of
        image ids.

        :param image_ids: list of ids
        :return: list of file paths
        """

        return [os.path.join(self._image_directory, image_id)
                for image_id in image_ids]

    def _get_mean(self):
        """
        Goes over all images and computes mean pixel value.

        :return: pixel mean
        """

        pixel_sum = pixel_count = 0
        image_file_paths = self._get_image_file_paths(self._image_rle_masks.keys())

        for image_file_path in image_file_paths:
            image = load_image(image_file_path, size=self._image_size) / 255
            pixel_sum += image.sum()
            pixel_count += image.size

        return pixel_sum / pixel_count

    def _get_image_mask(self, rle):
        """
        Creates a all-zeros mask or decodes non-null
        rle. The mask is converterd to a correct shape.

        :param rle: string with an rle mask
        :return: mask as a numpy array
        """

        if pd.isnull(rle):
            return np.zeros(shape=self._mask_size)

        return self._mask_downsampler.downsample(decode_rle(rle))

    def _generate_data(self, selected_image_ids):
        """
        Loads images given the image ids as a numpy array.

        :param selected_image_ids: list of image ids to load
        :return: tuple of two numpy arrays (images and labels)
        """

        image_file_paths = self._get_image_file_paths(selected_image_ids)

        # ravel to 1d
        masks = [self._get_image_mask(self._image_rle_masks[image_id]).ravel()
                 for image_id in selected_image_ids]

        images = [load_image(image_file_path, size=self._image_size)
                  for image_file_path in image_file_paths]

        # return x, y
        return (
            np.array(images).astype('float32') / 255,
            np.array(masks).astype('uint8'))
