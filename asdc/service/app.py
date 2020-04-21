#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import redis
from flask import Flask, request, jsonify
from .detect import process_request
from .constants import REDIS_HOST, REDIS_PORT


def create_app(redis_host=REDIS_HOST, redis_port=REDIS_PORT):
    """
    Creates a Flask app instance.

    :param redis_host: redis host
    :param redis_port: redis port

    :return: Flask app instance
    """

    app = Flask(__name__)
    db = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

    @app.route('/')
    def index():
        """
        Index page.
        """

        return 'ASDC Service'

    @app.route('/detect', methods=['POST'])
    def detect():
        """
        Detection API endpoint.
        """

        return jsonify(process_request(db, request.get_json()))

    return app
