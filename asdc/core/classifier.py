#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.models import Model  # , Sequential
# from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import (Dense, Activation, Conv2D,
                                            MaxPooling2D, Flatten, Dropout)


L2_LAMBDA = 0.01
DROPOUT = 0.25


class MaskDetection:
    """
    Class defines the neural model.
    """

    @staticmethod
    def create_model(image_shape, n_output):
        """
        Creates the model instance and compiles it.

        :param image_shape: tuple defining the shape of the input images
        :param n_output: number of output units
        """

        optimizer = RMSprop(lr=0.0001, decay=1e-6)
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
        m = Dense(512, kernel_regularizer=l2(L2_LAMBDA))(m)
        m = Activation('relu')(m)
        m = Dropout(0.5)(m)
        m = Dense(n_output)(m)
        o = Activation('sigmoid')(m)

        model = Model(i, o)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

#     @staticmethod
#     def _create_model_(image_shape, num_classes=NUM_CLASSES):
#         """
#         """

#         print('A')

#         optimizer = RMSprop(lr=0.0001, decay=1e-6)

#         if len(image_shape) == 2:
#             image_shape += (1,)

#         model = Sequential()
#         model.add(Conv2D(32, (3, 3), padding='same',
#                   input_shape=image_shape, kernel_regularizer=l2(0.01)))

#         model.add(Activation('relu'))
#         model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(0.01)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))

#         model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
#         model.add(Activation('relu'))
#         model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(0.01)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))

#         model.add(Flatten())
#         model.add(Dense(512, kernel_regularizer=l2(0.01)))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(num_classes))
#         model.add(Activation('softmax'))
#         model.compile(optimizer=optimizer,
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

#         return model
