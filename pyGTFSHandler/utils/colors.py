# -*- coding: utf-8 -*-
"""Deterministic color helpers shared by `models/routes.py` (filling in
missing `route_color`/`route_text_color`), `maps/` (badge/icon styling), and
`utils/plot_helpers.py`/example notebooks (the `CB_*` colorblind-safe
palette, for matplotlib/folium plots that need a handful of clearly
distinguishable categorical colors).

Kept dependency-free (no matplotlib/colorsys needed beyond the stdlib) so it
can be imported from the lightweight `models/routes.py` load path without
pulling in the optional `plot` extras.
"""

import colorsys
import hashlib

# A qualitative HSL palette: routes are assigned a hue deterministically
# (hashed from route_id) rather than a single hardcoded fallback color, so
# feeds with many routes missing route_color still end up visually distinct.
_SATURATION = 0.55
_LIGHTNESS = 0.5

# Colorblind-safe qualitative palette (Okabe & Ito, 2008) -- use this
# instead of matplotlib's default tab10 (or other hue-wheel palettes) for
# any plot/map that needs a handful of clearly distinguishable categorical
# colors. tab10's red/green and blue/purple pairs collapse under the
# common forms of color vision deficiency; every pair in this palette
# stays distinguishable under protanopia, deuteranopia, and tritanopia.
# Named individually (rather than only as a list) so call sites can pick
# specific, semantically-stable colors (e.g. always vermillion for
# "flagged"/warning) instead of depending on list order.
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_VERMILLION = "#D55E00"
CB_SKY = "#56B4E9"
CB_PURPLE = "#CC79A7"
CB_YELLOW = "#F0E442"
CB_BLACK = "#000000"

# Same eight colors as an ordered list, for `zip`-ing against categories
# (e.g. `dict(zip(sorted(shape_ids), COLORBLIND_SAFE_PALETTE))`). Blue
# first since it's the most universally distinguishable against a white
# background; black last since it doubles as this module's default
# text/outline color and is usually wanted as an accent, not a category
# fill.
COLORBLIND_SAFE_PALETTE = [
    CB_BLUE, CB_ORANGE, CB_GREEN, CB_VERMILLION, CB_SKY, CB_PURPLE, CB_YELLOW, CB_BLACK,
]


def route_id_to_color(route_id: str) -> str:
    """Hashes `route_id` to a deterministic, varied hex color (no leading
    `#`), by picking a hue around the color wheel and a fixed
    saturation/lightness so results stay readable as marker/badge fills."""
    digest = hashlib.md5(str(route_id).encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, _LIGHTNESS, _SATURATION)
    return f"{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def relative_luminance(hex_color: str) -> float:
    """WCAG-style perceived-luminance approximation (0-255 scale) for a hex
    color string (with or without leading `#`)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return 255.0
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b


def contrasting_text_color(hex_color: str) -> str:
    """Returns `"000000"` or `"ffffff"`, whichever contrasts better against
    `hex_color` as a background, using WCAG-ish relative luminance."""
    return "000000" if relative_luminance(hex_color) > 150 else "ffffff"
