#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import json
import numpy as np
from .constants import REQUEST_BATCH_SIZE, QUEUE_NAME
from ..core.utils import decode_image_b64


class RequestProcessor:
    """
    """

    def __init__(self, db, model):
        """
        """

        self._db = db
        self._model = model

    def _get_requests(self):
        """
        """

        return map(
            json.loads,
            self._db.lrange(QUEUE_NAME, 0, REQUEST_BATCH_SIZE - 1))

    @staticmethod
    def _create_model_input(requests):

        images, image_ids = zip(*((
            decode_image_b64(r['image_b64']), r['id']) for r in requests))

        return np.array(list(images)), list(image_ids)


