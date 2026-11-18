"""pyGTFSHandler: load, check and analyse GTFS transit feeds fast (Polars).

The entry point is :class:`Feed`: load one or several feeds, filter them by date, time, area or mode, and get headways
and speeds per stop. ``pyGTFSHandler.downloaders`` finds and downloads feeds (Mobility Database, Transitland, national
access points, ...).
"""
__version__ = '0.1.0'

from .feed import Feed

__all__ = ["Feed"]
