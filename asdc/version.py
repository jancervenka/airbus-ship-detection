#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import pkg_resources

__module_name__ = 'asdc'

try:
    __version__ = pkg_resources.get_distribution(__module_name__).version
except Exception:
    __version__ = 'unknown'
