#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import os
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from sklearn.model_selection import train_test_split
from .utils import get_image_rle_masks
from .constants import RLE_MASK_COL, IMAGE_ID_COL, IMAGE_SIZE, MASK_SIZE
from .classifier import MaskDetection
from .io import ImageBatchGenerator


class BalancedImageRleMasks:
    """
    Class containing functions for downsampling
    and balancing the dataset.
    """

    @staticmethod
    def _sample_class(image_rle_masks, select_positive, sample_size):
        """
        Selects one class from `image_rle_masks`.

        :param image_rle_masks: dataframe containing an image mask for each id
        :param select_positive: positive (non-zero) masks are selected
                                if `True`, otherwise negative
        :sample_size: number of class samples to select
        :return image_masks: filtered `image_masks`
        """

        row_mask = image_rle_masks[RLE_MASK_COL].isna()
        if select_positive:
            row_mask = ~row_mask

        return image_rle_masks[row_mask].sample(sample_size)

    @classmethod
    def _balance(cls, image_rle_masks, sample_size, pos_share):
        """
        Downsamples and balances `image_rle_masks`.

        :param image_rle_masks: dataframe containing an image mask for each id
        :param sample_size: size of the downsampled set
        :return: balanced and downsampled `image_masks`
        """

        positive_samples = round(sample_size * pos_share)
        negative_samples = sample_size - positive_samples

        balanced_image_masks = [
            cls._sample_class(image_rle_masks, True, positive_samples),
            cls._sample_class(image_rle_masks, False, negative_samples)]

        return pd.concat(balanced_image_masks, ignore_index=True)

    @staticmethod
    def _split(image_rle_masks):
        """
        Splits `image_rle_masks` to training, validation, and test sets.

        :param image_rle_masks: dataframe containing an image mask for each id
        :return: tuple of three dataframes containing the
                 training, validation, and test sets
        """

        train, test = train_test_split(image_rle_masks)
        train, val = train_test_split(train)
        return train, val, test

    @staticmethod
    def _to_dict(image_rle_masks):
        """
        Converts `image_rle_masks` dataframe to a dictionary

        :param image_rle_masks: dataframe containing an image mask for each id
        :return: `image_rle_masks` as a dictionary
        """

        return image_rle_masks.set_index(IMAGE_ID_COL).to_dict()[RLE_MASK_COL]

    @classmethod
    def create(cls, image_rle_masks, sample_size, pos_share=0.5, as_dict=True):
        """
        Downsamples `image_rle_masks`, balance the classes
        and splits the data into training, validation, and test.

        :param image_rle_masks: dataframe containing an image mask for each id
        :param sample_size: size of the downsampled set
        :param pos_share: share of the non-zero masks
        :param as_dict: if `True` result is returned as a tuple of dictionaries
        :return: tuple of three dataframes or dictionaries containing the
                 training, validation, and test image masks
        """

        splits = cls._split(cls._balance(image_rle_masks, sample_size, pos_share))

        if as_dict:
            return tuple(map(cls._to_dict, splits))
        else:
            return splits


class Pipeline:
    """
    Class containing model trainig pipeline.
    """

    def __init__(self, cfg):
        """
        Initiates the class.

        :param cfg: loaded config in `configparser.ConfigParser` instance
        """

        np.random.seed(0)
        self._cfg = cfg

    def _init_logger(self):
        """
        Initiates the logger.
        """

        log_file_name = '{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        log_directory = self._cfg.get('common', 'log_directory')
        log_file_path = os.path.join(log_directory, log_file_name)
        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO,
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])

    def _get_image_rle_masks(self):
        """
        """

        return get_image_rle_masks(
            pd.read_csv(self._cfg.get('data', 'ground_truth_file')))

    def _init_generator(self, image_masks):
        """
        """

        return ImageBatchGenerator(
            image_directory=self._cfg.get('data', 'image_directory'),
            image_rle_masks=image_masks,
            mask_size=MASK_SIZE,
            image_size=IMAGE_SIZE)

    def _train(self):
        """
        """

        logging.info('Hello')

        xy_train, xy_val, xy_test = BalancedImageRleMasks.create(
            image_rle_masks=self._get_image_rle_masks(),
            sample_size=self._cfg.getint('training', 'sample_size'),
            pos_share=0.5)

        generator_train = self._init_generator(xy_train)
        generator_val = self._init_generator(xy_val)

        logging.info('Generators ready.')

        # TODO: change model to sigmoid/binary cross entropy
        # TODO: pos share from config
        n_output = MASK_SIZE[0] ** 2
        model = MaskDetection._create_model(image_shape=(128, 128, 3), n_output=n_output)
        model.fit_generator(
            generator=generator_train,
            validation_data=generator_val,
            use_multiprocessing=True,
            workers=mp.cpu_count(),
            verbose=1,
            epochs=15)
        model.save('/home/honza/airbus-ship-detection/model_files/asdc_1_2000.h5')

    def run(self):
        """
        """

        self._init_logger()
        self._train()
