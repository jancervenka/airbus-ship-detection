#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import argparse
from .training import Pipeline
from .. import version


def _get_main_argparser():
    """
    Creates the CLI argument parser.
    """

    parser = argparse.ArgumentParser(
        conflict_handler='resolve',
        description='asdc',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-v', '--version', action='version',
                        version='version: "{0}"'.format(version.__version__))

    parser.add_argument('-c', '--config', type=str, required=True,
                        dest='config', help='Path to the config file.')

    return parser


def run():
    """
    Runs the entry point
    """

    args = _get_main_argparser().parse_args()

    Pipeline(args).run()


if __name__ == '__main__':
    run()
