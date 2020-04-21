#!/usr/bin/env bash
# -*- coding: utf-8 -*-

echo "Starting ASDC service..."

python -m asdc.main backend -m model &
python -m asdc.main app &
