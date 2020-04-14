#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from .constants import NUM_CLASSES
from .utils import resize_image


def load_image(image_file_path, scale=None):
    """
    Loads an image.

    :param image_file_path: path to the image
    :param scale: if not `None`, the image is resized
           using this scale
    :return: image as numpy array
    """

    image = cv2.imread(image_file_path)
    if image is None:
        raise FileNotFoundError(f'Image {image_file_path} not found.')

    if scale is not None:
        image = resize_image(image, scale)

    # change volume (n, m) to (n, m, 1)
    return np.expand_dims(image, axis=2) if len(image.shape) == 2 else image


class ImageBatchGenerator(Sequence):
    """
    Batch image generator. Iterable where each item
    contains one batch of images and labels (as a tuple
    of two numpy arrays). Each pass through the generator
    (triggered by `self.on_epoch_end`) reshuffles the order
    of images in the batches.

    Batches are loaded on demand (lazy iteration).

    This generator supports parallel processing.
    """
    # TODO: use mean image

    def __init__(self, image_directory, image_labels,
                 batch_size=32, shuffle=True, image_scale=None):
        """
        Initiates the class.

        :param image_directory: path to the image directory
        :param image_labels: dictionary `{image_id: label,}`
                             containing all images that should
                             be included in the generator.
        :param batch_size: size of one image batch
        :param shuffle: if `True`, reshuflles image id indexes
                        making different image/batch order for each epoch
        :param image_scale: if not `None`, the images are resized
                            using the scale value
        """

        self._image_directory = image_directory
        self._image_labels = image_labels
        self._image_ids = list(image_labels.keys())
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._image_scale = image_scale
        self.on_epoch_end()

    def __len__(self):
        """
        Computes generator length in number of batches.
        (one epoch = one pass through the generator)

        :return: length
        """

        return int(np.ceil(len(self._image_labels) / self._batch_size))

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
        image_file_paths = self._get_image_file_paths(self._image_labels.keys())

        for image_file_path in image_file_paths:
            image = load_image(image_file_path, scale=self._image_scale) / 255
            pixel_sum += image.sum()
            pixel_count += image.size

        return pixel_sum / pixel_count

    def _generate_data(self, selected_image_ids):
        """
        Loads images given the image ids as a numpy array.

        :param selected_image_ids: list of image ids to load
        :return: tuple of two numpy arrays (images and labels)
        """

        image_file_paths = self._get_image_file_paths(selected_image_ids)

        labels = [self._image_labels[image_id] for image_id in selected_image_ids]
        images = [load_image(image_file_path, scale=self._image_scale)
                  for image_file_path in image_file_paths]

        # return x, y
        return (
            np.array(images).astype('float32') / 255,
            to_categorical(labels, num_classes=NUM_CLASSES))
