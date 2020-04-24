#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import redis
import multiprocessing as mp
from .app import create_app
from .backend import RequestProcessor
from .constants import (REDIS_HOST, REDIS_PORT,
                        DEFAULT_SERVICE_HOST, DEFAULT_SERIVCE_PORT)


def run_app(args):
    """
    Runs the API frontend.
    """

    app = create_app(
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT)

    app.run(
        host=DEFAULT_SERVICE_HOST,
        port=DEFAULT_SERIVCE_PORT,
        debug=args.debug)


def run_backend(args):
    """
    Runs the request processing backend.
    """

    # load_model(args.model)
    from ..tests.utils import MockKerasModel
    model = MockKerasModel(image_shape=(128, 128, 3), n_output=1)

    db = redis.StrictRedis(
        host=REDIS_HOST, port=REDIS_PORT, db=0)

    request_processor = RequestProcessor(db=db, model=model)
    request_processor.run()


def run_service(args):
    """
    Runs the app and the backend as two processed.
    """

    process_backend = mp.Process(target=run_backend, args=(args,))
    process_backend.start()

    process_app = mp.Process(target=run_app, args=(args,))
    process_app.start()
    # process_app.join()
