#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import logging
import functools
import numpy as np
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import ParameterSampler, KFold
from .io import ImageBatchGenerator
from .utils import image_rle_masks_to_dict


class RandomSearch:
    """
    Class providing cross-validate random search
    hyperparameter optimization.
    """

    def __init__(self, model_factory, image_directory, image_shape,
                 param_grid, n_param_samples=5, cv=5, epochs=2, n_jobs=1,
                 verbose_training=False):
        """
        Initiates the class.

        :param model_factory: function genereting model instances
        :param image_directory: path to the image directory
        :param image_shape: tuple defining the shape of the model
                            input images; `(width, height, dephth)`
        :param param_grid: dictionary containing the hyperparameter
                           grid; `{'param_1': [value_1, value_2]}`
        :param n_param_samples: number of samples to draw from the
                                hyperparamater distributions
        :param cv: number of cross-validation folds
        :param epochs: number of training epochs for each evaluation
        :param n_jobs:  number of processes to used for the training
        :param verbose_training: if `True`, the training is verbose
        """

        self._model_factory = functools.partial(
            model_factory, image_shape=image_shape, n_output=1)

        self._folds = KFold(n_splits=cv)
        self._epochs = epochs
        self._image_shape = image_shape
        self._image_directory = image_directory
        self._n_jobs = n_jobs
        self._use_multiprocessing = n_jobs > 1
        self._verbose_training = verbose_training

        self._param_samples = ParameterSampler(param_grid, n_iter=n_param_samples)
        self.results = None
        self.best_params = None

    def _init_generator(self, image_rle_masks, ix):
        """
        Initiates a training or validation generator using
        a subset of the `image_rle_masks` dataframe.

        :param image_rle_masks: dataframe containing image
                                ids and their masks
        :param ix: list of row indexes defining the
                   training/validation subset
        """

        mask_subset = image_rle_masks_to_dict(image_rle_masks.iloc[ix, :])
        return ImageBatchGenerator(image_directory=self._image_directory,
                                   image_rle_masks=mask_subset,
                                   mask_size=(1, 1),
                                   image_size=self._image_shape[:2])

    def _train_model(self, image_rle_masks, ix_train, ix_val, params):
        """
        Trains a model on the supplied data and hyperparameters.

        :param image_rle_masks: dataframe containing image
                                ids and their masks
        :param ix_train: list of row indexes defining the
                         training set
        :param ix_val: list of row indexes defining the
                       validation set
        :param params: dictionary containing the model
                       hyperparameters

        :return: achieved loss
        """

        model = self._model_factory(**params)
        loss_ix = model.metrics_names.index('loss')
        model.fit_generator(
            generator=self._init_generator(image_rle_masks, ix_train),
            verbose=int(self._verbose_training),
            epochs=self._epochs,
            use_multiprocessing=self._use_multiprocessing,
            workers=self._n_jobs)

        final_loss = model.evaluate_generator(
            self._init_generator(image_rle_masks, ix_val))[loss_ix]

        logging.info(f'Loss={final_loss:.6f} for params={params}.')
        clear_session()
        return final_loss

    def _eval_params(self, image_rle_masks, params):
        """
        Cross-validates `params` model hyperparameters.

        :param image_rle_masks: dataframe containing image
                                ids and their masks
        :param params: dictionary containing the model
                       hyperparameters
        :return: dictionary containing the achieved losses from
                 each cross-validation fold as well as the overall
                 mean loss;
        """

        losses = [self._train_model(image_rle_masks, ix_train, ix_val, params)
                  for ix_train, ix_val in self._folds.split(image_rle_masks)]

        return {'mean_loss': np.mean(losses), 'params': params, 'losses': losses}

    def fit(self, image_rle_masks):
        """
        Evaluates the random search. Results (loss for each parameter
        sample) are stored as list of dictionaries in `self.results`.
        Best parameters are stored in `self.best_params` as a dictionary.

        :param image_rle_masks: dataframe containing image
                                ids and their masks
        """

        logging.info(
            f'Fitting n='
            f'{self._folds.get_n_splits(image_rle_masks) * len(self._param_samples)}'
            f' models.')

        # worker = functools.partial(self._eval_params, image_rle_masks=image_rle_masks)
        # results = pool.map(worker, self._param_samples)

        results = [self._eval_params(image_rle_masks, params)
                   for params in self._param_samples]

        self.best_params = min(results, key=lambda x: x.get('mean_loss'))['params']
        self.results = results
