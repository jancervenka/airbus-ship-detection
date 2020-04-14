#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import os
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
from .utils import get_image_labels
from .constants import NUM_CLASSES, IMAGE_LABEL_COL, IMAGE_ID_COL, SCALE
from .classifier import ShipDetection
from .io import ImageBatchGenerator


class BalancedImageLabels:
    """
    Class containing functions for downsampling
    and balancing `image_labels`.
    """

    @staticmethod
    def _get_class_image_labels(image_labels, label, sample_size):
        """
        Selects one class from `image_labels`.

        :param image_labels: dataframe containing image label for each id
        :param label: selected class
        :sample_size: number of class samples to select
        :return image_labels: filtered `image_labels`
        """

        mask = image_labels[IMAGE_LABEL_COL] == label
        return image_labels[mask].sample(sample_size)

    @classmethod
    def _balance(cls, image_labels, sample_size):
        """
        Downsamples and balances `image_labels`.

        :param image_labels: dataframe containing image label for each id
        :param sample_size: size of the downsampled set
        :return: balanced and downsampled `image_labels`
        """

        class_samples = sample_size // NUM_CLASSES
        balanced_image_labels = [
            cls._get_class_image_labels(image_labels, label, class_samples)
            for label in [0, 1]]

        return pd.concat(balanced_image_labels, ignore_index=True)

    @staticmethod
    def _split(image_labels):
        """
        Splits `image_labels` to training, validation, and test sets.

        :param image_labels: dataframe containing image label for each id
        :return: tuple of three dataframes containing the
                 training, validation, and test sets
        """

        train, test = train_test_split(image_labels)
        train, val = train_test_split(train)
        return train, val, test

    @staticmethod
    def _to_dict(image_labels):
        """
        Converts `image_labels` dataframe to a dictionary

        :param image_labels: dataframe containing image label for each id
        :return: `image_labels` as a dictionary
        """

        return image_labels.set_index(IMAGE_ID_COL).to_dict()[IMAGE_LABEL_COL]

    @classmethod
    def create(cls, image_labels, sample_size, as_dict=True):
        """
        Downsamples `image_labels`, balance the classes
        and splits the data into training, validation, and test.

        :param image_labels: dataframe containing image label for each id
        :param sample_size: size of the downsampled set
        :param as_dict: if `True` result is returned as a
                        tuple of dictionaries
        :return: tuple of three dataframes or dictionaries
                 containing the training, validation, and test image labels
        """

        splits = cls._split(cls._balance(image_labels, sample_size))

        if as_dict:
            return tuple(map(cls._to_dict, splits))
        else:
            return splits


class Pipeline:
    """
    Class containing model trainig pipeline.
    """

    def __init__(self, args):
        """
        Initiates the class.

        :param args: `argparse.ArgumentParser` instance
        """

        np.random.seed(0)

        self._args = args
        self._cfg = ConfigParser()
        self._cfg.read(self._args.config)

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

    def _get_image_labels(self):
        """
        """

        return get_image_labels(
            pd.read_csv(self._cfg.get('data', 'ground_truth_file')))

    def _init_generator(self, image_labels):
        """
        """

        return ImageBatchGenerator(
            image_directory=self._cfg.get('data', 'image_directory'),
            image_labels=image_labels,
            image_scale=SCALE)

    def _train(self):
        """
        """

        logging.info('Hello')

        labels_train, labels_val, label_test = BalancedImageLabels.create(
            image_labels=self._get_image_labels(),
            sample_size=self._cfg.getint('training', 'sample_size'))

        generator_train = self._init_generator(labels_train)
        generator_val = self._init_generator(labels_val)

        logging.info('Generators ready.')

        model = ShipDetection._create_model(image_shape=(128, 128, 3))
        model.fit_generator(
            generator=generator_train,
            validation_data=generator_val,
            use_multiprocessing=True,
            workers=mp.cpu_count(),
            verbose=1,
            epochs=20)
        model.save('/home/honza/airbus-ship-detection/model_files/asdc.h5')

    def run(self):
        """
        """

        self._init_logger()
        self._train()
