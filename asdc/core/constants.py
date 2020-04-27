#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

DEFAULT_IMAGE_SIZE = (768, 768)

IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
MASK_SIZE = (1, 1)

IMAGE_ID_COL = 'ImageId'
RLE_MASK_COL = 'EncodedPixels'

MODEL_FILE_NAME = 'asdc_{}.h5'
MODEL_HISTORY_FILE_NAME = 'asdc_{}.json'
