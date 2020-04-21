#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

from configparser import ConfigParser
from .training import Pipeline

# TODO: rename core to training?


def _get_config(args):
    """
    Loads config file

    :param args: `argparse.ArgumentParser` instance
    :return: `configparser.ConfigParser` instance
    """

    cfg = ConfigParser()
    cfg.read(args.config)
    return cfg


def run_training(args):
    """
    Runs the entry point
    """

    cfg = _get_config(args)
    Pipeline(cfg).run()
