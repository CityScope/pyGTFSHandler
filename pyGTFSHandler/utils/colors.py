# -*- coding: utf-8 -*-
"""Deterministic color helpers shared by `models/routes.py` (filling in
missing `route_color`/`route_text_color`) and `maps/` (badge/icon styling).

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
