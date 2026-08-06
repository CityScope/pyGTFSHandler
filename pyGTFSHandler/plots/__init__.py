# -*- coding: utf-8 -*-
"""Matplotlib plotting helpers shared by `examples/*.ipynb`.

Kept out of the notebooks themselves so a notebook reads as "how to use
`pyGTFSHandler`" -- filtering, computing headway, inspecting
`direction_id` -- rather than as a pile of incidental matplotlib
boilerplate (axis limits, annotation offsets, legend placement) that has
nothing to do with the package's own API. Every function here takes
plain data (dicts of coordinates/bearings/colors, already computed by the
notebook via the real `pyGTFSHandler` API) and an `Axes`/`Figure` to draw
into -- none of them call into `pyGTFSHandler` themselves, so they stay
reusable across different example feeds.
"""
