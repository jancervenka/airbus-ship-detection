#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import cv2
import time
import json
import numpy as np
from .constants import REQUEST_BATCH_SIZE, QUEUE_NAME, BACKEND_WAITING_SEC
from ..core.utils import decode_image_b64, check_image_rgb


class RequestProcessor:
    """
    Backend class for batch processing of incoming requests.
    """

    def __init__(self, db, model):
        """
        Initiates the class.

        :param db: Redis connection pool
        :param model: Keras model instance
        """

        self._db = db
        self._model = model

    def _get_requests(self):
        """
        Gets new requests from the Redis queue that need
        to be processed.

        :return: list of queue values (dictionaries
                 `{'id': request_id, 'image_b64': image_b64'}`)
        """

        return list(map(
            json.loads,
            self._db.lrange(QUEUE_NAME, 0, REQUEST_BATCH_SIZE - 1)))

    def _resize_if_required(self, image):
        """
        Resizes an image if the size is not compatible with
        the model input.

        :param image: numpy array containing the image
        :return: numpy array containing the original or
                 resized image
        """

        # TODO: would padding be better than resizing?

        if image.shape != self._model.input_shape[1:]:
            w, h = self._model.input_shape[1:3]
            return cv2.resize(image,  (w, h))

        return image

    def _create_image(self, image_b64):
        """
        Decodes base64 image string, checks channel
        compatibility, resizes if necessary, converts
        to (0, 1) value range.

        :param image_b64: base64-encoded image
        :return: numpy array containing the image or
                 `None` (failure)
        """

        try:
            image = decode_image_b64(image_b64)
            check_image_rgb(image)

            image = self._resize_if_required(image)
            return image.astype('float32') / 255

        except Exception:
            return None

    def _prepare_ok_nok_requests(self, requests):
        """
        Tries to decode and convert images contained
        in the `requests`. Creates two iterables, one
        contains successfully prepared requests, the
        second one containes requests that cannot be
        processed.

        :param requests: list of request dictionaries
        :return: tuple of two iterables `(ok, nok)`
        """

        images_ids = (
            (self._create_image(r['image_b64']), r['id']) for r in requests)

        ok = filter(lambda x: x[0] is not None, images_ids)
        nok = filter(lambda x: x[0] is None, images_ids)

        return ok, nok

    def _process_nok_requests(self, nok_requests):
        """
        Sends error results to the Redis store for requests
        that cannot be proccessed.

        :param nok_requests: list of request tuples
        """

        for _, request_id in nok_requests:
            self._db.set(request_id, json.dumps({'error': 'image_not_compatible'}))

    def _process_ok_requests(self, ok_requests):
        """
        Makes prediction on the images in the `ok_requests` and
        sends the results to the Redis store for consumption.

        :param nok_requests: list of request tuples
        """

        # unziping makes always two lists of the same length
        ok_requests = list(ok_requests)
        if ok_requests:
            images, request_ids = zip(*ok_requests)
            images = np.array(images)

            # TODO: use rle to encode the mask
            # TODO: predict class? threshold the results?
            predictions = self._model.predict(images).any(axis=1).tolist()
            for prediction, request_id in zip(predictions, request_ids):
                self._db.set(request_id, json.dumps({'prediction': float(prediction)}))

    def _pop_queue(self, n):
        """
        Removes `n` requests from the front of the Redis queue.

        :param n: number of requests to be removed
        """
        self._db.ltrim(QUEUE_NAME, n, -1)

    def _process_one_batch(self):
        """
        Process one batch of requests.
        """

        requests = self._get_requests()
        ok_requests, nok_requests = self._prepare_ok_nok_requests(requests)
        self._process_nok_requests(nok_requests)
        self._process_ok_requests(ok_requests)

        self._pop_queue(len(requests))

    def run(self):
        """
        Runs the backend processing.
        """

        while True:
            self._process_one_batch()
            time.sleep(BACKEND_WAITING_SEC)
