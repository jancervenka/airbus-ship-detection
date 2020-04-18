#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import time
import json
import traceback
from uuid import uuid4
from .constants import QUEUE_NAME, REQUEST_TIMEOUT_SEC, REQUEST_WAITING_SEC


def store_push(db, image_b64):
    """
    Pushes an encoded image with an assigned uuid
    to a Redis queue.

    :param db: Redis connection pool
    :param image_b64: base64-encoded image
    :return: uuid identifying the request used
             for result retrieval
    """

    request_id = str(uuid4())
    db.rpush(
        QUEUE_NAME, json.dumps({'id': request_id, 'image_b64': image_b64}))

    return request_id


def store_get(db, request_id):
    """
    Attempts to retrieve a processed request
    from Redis store. The retrieval must happen
    within the request timeout.

    :param db: Redis connection pool
    :param request_id: uuid identifying the request

    :return: dictionary containing the result or an error
    """

    t_0 = time.time()
    while (time.time() - t_0) < REQUEST_TIMEOUT_SEC:
        result = db.get(request_id)

        if result is not None:
            db.delete(request_id)
            return json.loads(result.decode())

        else:
            time.sleep(REQUEST_WAITING_SEC)

    return {'error': 'request_timeout'}


def process_request(db, payload):
    """
    Sends the request over Redis to be processed by
    the backend. Then attempts to get the result back.

    :param db: Redis connection pool
    :param payload: request json payload containing
                    the encoded image

    :return: dictionary containing the result or an error
    """

    if payload:
        image_b64 = payload.get('image', None)
    else:
        return {'error': 'no_json_data'}

    try:
        request_id = store_push(db, image_b64)
        return store_get(db, request_id)

    except Exception:

        return {'error': traceback.format_exc()}
