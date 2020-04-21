#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import argparse
import asdc.core
import asdc.service
from . import version


def _create_arg_parser():
    """
    Creates the CLI argument parser.
    """

    parser = argparse.ArgumentParser(conflict_handler='resolve',
                                     description='asdc')

    parser.add_argument('-v', '--version', action='version',
                        version='version: "{0}"'.format(version.__version__))

    subparsers = parser.add_subparsers()
    parser_app = subparsers.add_parser('app')
    parser_app.add_argument('-d', '--debug', action='store_const',
                            default=False, const=True, dest='debug',
                            help='app debug mode')

    parser_backend = subparsers.add_parser('backend')
    parser_backend.add_argument('-m', '--model', type=str, required=True,
                                dest='model', help='Path to the model h5 file.')

    parser_training = subparsers.add_parser('training')
    parser_training.add_argument('-c', '--config', type=str, required=True,
                                 dest='config', help='path to the config file')

    parser_app.set_defaults(func=asdc.service.run_app)
    parser_backend.set_defaults(func=asdc.service.run_backend)
    # parser_backend.set_defaults(func=lambda args: print('hello backend, model', args.model))
    parser_training.set_defaults(func=asdc.core.run_training)

    return parser


def main():
    """
    Runs ASDC system.
    """

    args = _create_arg_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
