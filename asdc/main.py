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
    parser_service = subparsers.add_parser('service')
    parser_service.add_argument('-d', '--debug', action='store_const',
                                default=False, const=True, dest='debug',
                                help='app debug mode')
    parser_service.add_argument('-m', '--model', type=str, required=True,
                                dest='model', help='Path to the model h5 file.')

    parser_training = subparsers.add_parser('training')
    parser_training.add_argument('-c', '--config', type=str, required=True,
                                 dest='config', help='path to the config file')

    parser_service.set_defaults(func=asdc.service.run_service)
    parser_training.set_defaults(func=asdc.core.run_training)

    return parser


def main():
    """
    Runs the ASDC system.
    """

    args = _create_arg_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
