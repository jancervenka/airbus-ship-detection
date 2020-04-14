#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Dense, Activation, Conv2D,
                                     MaxPooling2D, Flatten, Dropout)


NUM_CLASSES = 2


class ShipDetection:
    """
    Class for binary image classification.
    """

    @staticmethod
    def _create_model(image_shape, num_classes=NUM_CLASSES):
        """
        Creates a neural model.

        :param image_shape: shape of the input images
        :param num_classes: number of classes in the training set
        :return: compiled keras model
        """

        optimizer = RMSprop(learning_rate=0.0001, decay=1e-6)

        if len(image_shape) == 2:
            image_shape += (1,)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                  input_shape=image_shape, kernel_regularizer=regularizers.l2(0.01)))

        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
