#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import (Dense, Activation, Conv2D,
                                            MaxPooling2D, Flatten, Dropout)


L2_LAMBDA = 0.01
DROPOUT = 0.25


class MaskDetection:
    """
    Class defines the neural model.
    """
    # TODO: callbacks
    # TODO: lambda layer for zero mean?

    @staticmethod
    def create_model(image_shape, n_output, n_dense=512, lr=0.0001):
        """
        Creates the model instance and compiles it.

        :param image_shape: tuple defining the shape of the input images
        :param n_output: number of output units
        :param n_dense: neumber of units in the dense layer
        :param lr: optimizer learning rate

        :return: compiled model instance
        """

        optimizer = RMSprop(lr=lr, decay=1e-6)
        image_shape = image_shape + (3,) if len(image_shape) == 2 else image_shape

        i = m = Input(shape=image_shape)

        for n_filters in (32, 64):
            m = Conv2D(n_filters, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA))(m)
            m = Activation('relu')(m)
            m = Conv2D(n_filters, (3, 3), kernel_regularizer=l2(L2_LAMBDA))(m)
            m = Activation('relu')(m)
            m = MaxPooling2D(pool_size=(2, 2))(m)
            m = Dropout(DROPOUT)(m)

        m = Flatten()(m)
        m = Dense(n_dense, kernel_regularizer=l2(L2_LAMBDA))(m)
        m = Activation('relu')(m)
        m = Dropout(0.5)(m)
        m = Dense(n_output)(m)
        o = Activation('sigmoid')(m)

        model = Model(i, o)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model
