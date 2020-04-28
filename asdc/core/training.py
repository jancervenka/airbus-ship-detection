#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import os
import json
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import clear_session
from .classifier import MaskDetection
from .optimization import RandomSearch
from .io import ImageBatchGenerator
from .utils import (get_image_rle_masks, image_rle_masks_to_dict,
                    convert_history)
from .constants import (RLE_MASK_COL, IMAGE_SIZE, MASK_SIZE, IMAGE_SHAPE,
                        MODEL_FILE_NAME, MODEL_HISTORY_FILE_NAME)


class BalancedImageRleMasks:
    """
    Class containing functions for downsampling
    and balancing the dataset.
    """
    _to_dict = image_rle_masks_to_dict

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

    @classmethod
    def create(cls, image_rle_masks, sample_size, pos_share=0.5, as_dict=True, split=True):
        """
        Downsamples `image_rle_masks` and balance the classes.

        :param image_rle_masks: dataframe containing an image mask for each id
        :param sample_size: size of the downsampled set
        :param pos_share: share of the non-zero masks
        :param as_dict: if `True`, the result is converted to a dictionary
        :param split: if `True`, the result is split into training, test, validation
        :return: dictionary/dataframe or tuple (when `split=True`) of
                 dictionaries/dataframe containing the image masks
        """

        selected = cls._balance(image_rle_masks, sample_size, pos_share)
        if split:
            splits = cls._split(cls._balance(image_rle_masks, sample_size, pos_share))
            return tuple(map(cls._to_dict, splits)) if as_dict else splits

        else:
            return cls._to_dict(selected) if as_dict else selected


class Pipeline:
    """
    Class containing model trainig pipeline.
    """
    # TODO: image size mask size to config
    # TODO: assure training and random search sets are different
    # TODO: assure sample size is enough for cv/random search, batch size

    def __init__(self, cfg):
        """
        Initiates the class.

        :param cfg: loaded config in `configparser.ConfigParser` instance
        """

        np.random.seed(0)
        self._cfg = cfg
        self._n_jobs = None
        self._training_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _init_logger(self):
        """
        Initiates the logger.
        """

        log_file_name = '{}.log'.format(self._training_datetime)
        log_directory = self._cfg.get('common', 'log_directory')
        log_file_path = os.path.join(log_directory, log_file_name)
        os.makedirs(log_directory, exist_ok=True)

        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO,
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])

    def _get_image_rle_masks(self):
        """
        Loads `ground_truth` and create image RLE masks from it.

        :return: dataframe containing image ids and their masks
        """

        logging.info('Loading image masks.')

        return get_image_rle_masks(
            pd.read_csv(self._cfg.get('data', 'ground_truth_file')))

    def _init_generator(self, image_rle_masks):
        """
        Creates an `ImageBatchGenerator` instance.

        :param image_rle_masks: dictionary containing image ids and masks
                                that should be included in the generator
        :return: `ImageBatchGenerator` instance
        """

        return ImageBatchGenerator(
            image_directory=self._cfg.get('data', 'image_directory'),
            image_rle_masks=image_rle_masks,
            mask_size=MASK_SIZE,
            image_size=IMAGE_SIZE)

    def _find_best_params(self):
        """
        Runs the random search optimization.

        :return: `optimization.RandomSearch` instance containing
                 the best hyperparameters.
        """
        logging.info('Running random search.')

        image_rle_masks = BalancedImageRleMasks.create(
            self._get_image_rle_masks(),
            sample_size=self._cfg.getint('random_search', 'sample_size'),
            pos_share=self._cfg.getfloat('training', 'positive_share'),
            split=False,
            as_dict=False)

        random_search = RandomSearch(
            model_factory=MaskDetection.create_model,
            image_directory=self._cfg.get('data', 'image_directory'),
            image_shape=IMAGE_SIZE + (3, ),
            param_grid=json.loads(self._cfg.get('random_search', 'param_grid')),
            n_param_samples=self._cfg.getint('random_search', 'n_param_samples'),
            cv=self._cfg.getint('random_search', 'cv'),
            epochs=2,
            n_jobs=self._n_jobs)

        random_search.fit(image_rle_masks)

        logging.info(f'Random search complete, best_params={random_search.best_params}.')
        return random_search

    def _save_model(self, model):
        """
        Saves `model` to a file.

        :param model: model to save.
        """

        model_file_name = MODEL_FILE_NAME.format(self._training_datetime)
        model_file_path = os.path.join(
            self._cfg.get('data', 'model_directory'), model_file_name)
        model.save(model_file_path)
        logging.info(f'Model stored to {model_file_path}.')

    def _save_history(self, history):
        """
        Saves the model training history to a json file.

        :param history: model history dictionary
        """

        history_file_name = MODEL_HISTORY_FILE_NAME.format(
            self._training_datetime)
        history_file_path = os.path.join(
            self._cfg.get('data', 'model_directory'), history_file_name)

        with open(history_file_path, 'w') as f:
            json.dump(convert_history(history), f, indent=4)

        logging.info(f'History stored to {history_file_path}.')

    def _train(self):
        """
        Trains the model.
        """

        params = {}
        if self._cfg.getboolean('training', 'use_random_search'):
            params = self._find_best_params().best_params

        logging.info('Training the final model.')

        image_rle_mask_splits = BalancedImageRleMasks.create(
            image_rle_masks=self._get_image_rle_masks(),
            sample_size=self._cfg.getint('training', 'sample_size'),
            pos_share=self._cfg.getfloat('training', 'positive_share'))

        gen_train, gen_val, gen_test = [self._init_generator(xy)
                                        for xy in image_rle_mask_splits]

        logging.info('Generators ready.')
        clear_session()
        model = MaskDetection.create_model(
            image_shape=IMAGE_SHAPE, n_output=MASK_SIZE[0] ** 2, **params)

        model.fit_generator(
            generator=gen_train,
            validation_data=gen_val,
            use_multiprocessing=(self._n_jobs > 1),
            workers=self._n_jobs,
            verbose=1,
            callbacks=MaskDetection.create_callbacks(),
            epochs=self._cfg.getint('training', 'epochs'))

        # TODO: log evaluation
        model.evaluate_generator(gen_test)
        self._save_model(model)
        self._save_history(model.history.history)

    def _get_n_jobs(self):
        """
        Gets the number of workers to use for training.

        :return: number of jobs
        """

        if self._cfg.getboolean('common', 'multiprocessing'):
            n_jobs = mp.cpu_count()
        else:
            n_jobs = 1

        logging.info(f'Multiprocessing={n_jobs > 1}. Using n_jobs={n_jobs}.')

        return n_jobs

    def run(self):
        """
        Runs the training.
        """

        try:

            self._init_logger()
            logging.info('Hello from ASDC training!')
            self._n_jobs = self._get_n_jobs()
            self._train()
        finally:
            logging.shutdown()
