# -*- coding: utf-8 -*-
"""Downloaders for Spanish GTFS sources.

Currently holds `nap`, the client for Spain's National Access Point
(https://nap.transportes.gob.es), Spain's official catalog of public
transport datasets. Country-specific downloader packages (this one, and
any future ones) follow the same pattern: a `<country>/utils.py` module
for helpers specific to that country's source(s), and one module per
source implementing `downloaders.base.BaseGTFSDownloader`.
"""

from .nap import NAPDownloader

__all__ = ["NAPDownloader"]
