# -*- coding: utf-8 -*-
"""Matplotlib figures used by the `examples/*.ipynb` notebooks.

See `pyGTFSHandler.plots` for why these live here instead of inline in
the notebooks. Each function draws one figure/panel from already-computed
data; none of them touch a `Feed` or read a GTFS file.
"""

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..utils.colors import CB_BLACK, CB_BLUE, CB_PURPLE, CB_SKY, CB_VERMILLION, CB_YELLOW


def bearing_xy(deg):
    """Unit-circle `(x, y)` for a compass bearing (degrees from north)."""
    rad = math.radians(deg)
    return math.sin(rad), math.cos(rad)


def compute_headway(departures, start, end):
    """The library's own headway formula, reproduced standalone for the
    notebooks' worked examples -- see `analysis/stops.py`'s
    `get_headway_at_stops` for the real, integrated implementation."""
    d = np.sort(np.asarray(departures, dtype=float))
    g0 = (d[0] - start) + (end - d[-1])
    gaps = np.diff(d)
    return (g0 ** 2 + np.sum(gaps ** 2)) / (end - start)


def plot_network(ax, stop_coords, branches, title, lower_labels=frozenset(), legend=True, alpha=0.85, linewidth=3, lane_offset=0.0):
    """Draws one or more colored polylines through `stop_coords`, with a
    black dot + label at every stop. `branches`: `[(stop_id_path, color,
    label_or_None), ...]`. `lower_labels`: stop_ids whose label should be
    drawn below (rather than above) the point, for stops on a branch that
    dips below the trunk line.

    `lane_offset` (in the same units as `stop_coords`, e.g. degrees):
    when nonzero, each branch's line is nudged perpendicular to its own
    direction of travel by a per-branch amount -- otherwise two branches
    that traverse the exact same physical segment (e.g. a shape and its
    exact reverse) draw exactly on top of one another, and only the
    topmost color is ever visible.
    """
    n = len(branches)
    for branch_idx, (path, color, label) in enumerate(branches):
        xs = [stop_coords[s][0] for s in path]
        ys = [stop_coords[s][1] for s in path]
        if lane_offset:
            lane = branch_idx - (n - 1) / 2
            oxs, oys = [], []
            for i in range(len(xs)):
                dx = xs[min(i + 1, len(xs) - 1)] - xs[max(i - 1, 0)]
                dy = ys[min(i + 1, len(ys) - 1)] - ys[max(i - 1, 0)]
                seg_len = math.hypot(dx, dy) or 1.0
                perp_x, perp_y = -dy / seg_len, dx / seg_len
                oxs.append(xs[i] + perp_x * lane_offset * lane)
                oys.append(ys[i] + perp_y * lane_offset * lane)
            xs, ys = oxs, oys
        ax.plot(xs, ys, "-", color=color, linewidth=linewidth, solid_capstyle="round", label=label, zorder=2, alpha=alpha)

    seen = set()
    for path, _color, _label in branches:
        for sid in path:
            if sid in seen:
                continue
            seen.add(sid)
            lon, lat = stop_coords[sid]
            ax.scatter([lon], [lat], color=CB_BLACK, s=50, zorder=3)
            below = sid in lower_labels
            dy = -14 if below else 9
            va = "top" if below else "bottom"
            ax.annotate(sid, (lon, lat), textcoords="offset points", xytext=(0, dy), fontsize=10, fontweight="bold", ha="center", va=va)

    # Stop labels are drawn via `ax.annotate(..., textcoords="offset
    # points")`, which (like arrows) isn't included in matplotlib's
    # autoscale -- without an explicit margin, a stop near the data's own
    # top/bottom edge gets its label silently clipped at the axes border.
    all_lons = [lon for lon, _ in stop_coords.values()]
    all_lats = [lat for _, lat in stop_coords.values()]
    margin_lon = (max(all_lons) - min(all_lons)) * 0.12 or 0.001
    margin_lat = (max(all_lats) - min(all_lats)) * 0.12 or 0.001
    ax.set_xlim(min(all_lons) - margin_lon, max(all_lons) + margin_lon)
    ax.set_ylim(min(all_lats) - margin_lat, max(all_lats) + margin_lat)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    if legend and any(label for _, _, label in branches):
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)


def plot_headway_bar(ax, labels, values, colors, title, ylabel="headway (min)", ylim=None, fmt="{:.1f}"):
    """A simple annotated bar chart -- one bar per (label, value, color)."""
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_bearing_correction(fig, ax, bearings_by_shape, shape_colors, reconcile_fn):
    """One compass circle: every shape's raw `(fwd, bwd)` bearings (faint,
    dashed) and the same after `reconcile_fn` (`_reconcile_fwd_bwd`) forces
    them exactly 180 degrees apart (bold, solid) -- overlaid so the
    correction's effect (or lack of one) is visible directly, rather than
    split across two separate circles. `bearings_by_shape`: `{shape_id:
    (fwd, bwd)}`.
    """
    circle = plt.Circle((0, 0), 1, fill=False, color="lightgray", linewidth=1)
    ax.add_patch(circle)

    for shape_id, (fwd_raw, bwd_raw) in bearings_by_shape.items():
        color = shape_colors[shape_id]
        fwd, bwd = reconcile_fn(fwd_raw, bwd_raw)

        for deg, style in [(fwd_raw, "raw"), (bwd_raw, "raw"), (fwd, "corrected"), (bwd, "corrected")]:
            x, y = bearing_xy(deg)
            if style == "raw":
                ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=1.3, alpha=0.4, linestyle="dashed"))
            else:
                ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
                ax.annotate(f"{deg:.0f}°", (x * 1.15, y * 1.15), color=color, ha="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    handles = [plt.Line2D([0], [0], color=color, lw=2.5, label=shape_id) for shape_id, color in shape_colors.items()]
    handles += [
        plt.Line2D([0], [0], color="gray", lw=1.3, alpha=0.4, linestyle="dashed", label="raw"),
        plt.Line2D([0], [0], color="gray", lw=2.5, label="corrected"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.04))


def plot_widest_gap_circle(ax, corrected_by_shape, split, shape_colors, bin_color=None):
    """One compass circle: every shape's corrected `(fwd, bwd)` pair as
    arrows, the widest-gap `split` boundary as a dashed diameter, and the
    two resulting half-circles shaded by bin.
    """
    bin_color = bin_color or {0: CB_SKY, 1: CB_VERMILLION}
    circle = plt.Circle((0, 0), 1, fill=False, color="lightgray", linewidth=1)
    ax.add_patch(circle)

    split_rad = math.radians(split)
    for b in (0, 1):
        t0 = split_rad if b == 0 else split_rad + math.pi
        t1 = t0 + math.pi
        arc = np.linspace(t0, t1, 100)
        xs = np.concatenate([[0], np.sin(arc), [0]])
        ys = np.concatenate([[0], np.cos(arc), [0]])
        ax.fill(xs, ys, color=bin_color[b], alpha=0.15, zorder=0)

    for shape_id, angles in corrected_by_shape.items():
        color = shape_colors[shape_id]
        for a in angles:
            x, y = bearing_xy(a)
            ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
            ax.scatter([x], [y], color=color, s=70, zorder=4, edgecolor="white")

    sx, sy = bearing_xy(split)
    ax.plot([-sx * 1.3, sx * 1.3], [-sy * 1.3, sy * 1.3], "--", color=CB_BLACK, linewidth=1.5, zorder=1)
    ax.annotate(f"split = {split:.0f}°", (sx * 1.35, sy * 1.35), ha="center", fontsize=10, fontweight="bold")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=shape_id) for shape_id, color in shape_colors.items()]
    handles += [mpatches.Patch(color=bin_color[0], alpha=0.3, label="bin 0"), mpatches.Patch(color=bin_color[1], alpha=0.3, label="bin 1")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")


def plot_widest_gap_compass(ax, bearings_by_shape, title, split_fn, bins_fn):
    """A compass circle (N/E/S/W-labeled) with the widest-gap split
    boundary shaded, every shape's bearing(s) plotted and colored by its
    bin, and a small per-shape legend of exact bearings listed below the
    circle (rather than as inline point labels, which collide when two
    shapes are a fraction of a degree apart -- the whole point of the
    "spurious split" case this is built for).

    `bearings_by_shape`: `{shape_id: [angle_degrees, ...]}`.
    `split_fn`/`bins_fn`: the widest-gap split/bin functions to apply
    (`models.shapes._split_angle` / `_widest_gap_split`).
    """
    bins = bins_fn(bearings_by_shape)
    split = split_fn(bearings_by_shape)
    colors = {0: CB_BLUE, 1: CB_VERMILLION}

    if split is not None:
        wedge0 = mpatches.Wedge((0, 0), 1, 90 - split, 90 - split + 180, facecolor=colors[0], alpha=0.10, zorder=0)
        wedge1 = mpatches.Wedge((0, 0), 1, 90 - split - 180, 90 - split, facecolor=colors[1], alpha=0.10, zorder=0)
        ax.add_patch(wedge0)
        ax.add_patch(wedge1)
        for boundary_deg in (split, split + 180):
            bx, by = bearing_xy(boundary_deg)
            ax.plot([-bx, bx], [-by, by], color="gray", linestyle="--", linewidth=1, zorder=1)

    circle = plt.Circle((0, 0), 1, fill=False, color="lightgray", linewidth=1, zorder=1)
    ax.add_patch(circle)

    for compass_deg, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        cx, cy = bearing_xy(compass_deg)
        ax.text(cx * 1.18, cy * 1.18, label, color="darkgray", fontsize=8, ha="center", va="center", zorder=1)

    legend_lines = []
    for shape_id, angles in bearings_by_shape.items():
        b = bins[shape_id]
        for deg in angles:
            x, y = bearing_xy(deg)
            ax.scatter([x], [y], color=colors[b], s=80, zorder=3, edgecolor="white", linewidth=0.7)
        deg_strs = "/".join(f"{d:.1f}°" for d in angles)
        legend_lines.append((shape_id, deg_strs, colors[b]))

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4 - 0.16 * len(legend_lines), 1.4)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    y0 = -1.55
    for i, (shape_id, deg_strs, color) in enumerate(legend_lines):
        ax.text(-1.35, y0 - i * 0.16, f"● {shape_id}: {deg_strs}", color=color, fontsize=7.5, va="top", ha="left")


def plot_sector_split(ax, bearings_by_label, colors_by_label, split, n_divisions, title, show_labels=False, arrow_scale=1.3, legend=False, direction_ids=None, legend_colors=None):
    """One compass circle divided into `2*n_divisions` shaded sectors,
    with one arrow per `(label, bearing_degrees)` in `bearings_by_label`.

    Two labels landing on (nearly) the same bearing -- e.g. a shape and
    its exact reverse, which always produce the identical corrected
    bearing *pair*, just forward/backward swapped -- are staggered to
    different radii so both stay visible, rather than one silently
    hiding the other.

    `legend`: draws a color key below the circle, from `legend_colors`
    if given (one entry per logical group, e.g. per shape_id) or from
    `colors_by_label` otherwise (one entry per point -- noisier when
    `bearings_by_label` has multiple points per shape).
    `direction_ids`: optional `{label: id}` (e.g. each shape's resulting
    `direction_id`/sector) -- when given, printed as a line of text below
    the circle (and below the legend, if both are shown).
    """
    n_sectors = 2 * n_divisions
    boundary = split + (90 / n_divisions if n_divisions % 2 == 0 else 0)
    sector_colors = [CB_SKY, CB_PURPLE, CB_YELLOW, CB_BLACK]

    for i in range(n_sectors):
        t0 = math.radians(boundary + i * 360 / n_sectors)
        t1 = math.radians(boundary + (i + 1) * 360 / n_sectors)
        arc = np.linspace(t0, t1, 60)
        xs = np.concatenate([[0], np.sin(arc), [0]])
        ys = np.concatenate([[0], np.cos(arc), [0]])
        ax.fill(xs, ys, color=sector_colors[i % len(sector_colors)], alpha=0.25, zorder=0)

    circle = plt.Circle((0, 0), 1, fill=False, color="lightgray", linewidth=1)
    ax.add_patch(circle)

    # Stagger radii for labels sharing (nearly) the same angle: each new
    # label landing in an existing angle group gets the *next* radius
    # down, not the same one as whoever's already there.
    radius_by_label = {}
    group_reps = []  # representative angle of each group seen so far
    group_counts = []  # how many labels already assigned in that group
    for label, angle in bearings_by_label.items():
        group = next((i for i, a in enumerate(group_reps) if abs(((angle - a + 180) % 360) - 180) < 0.5), None)
        if group is None:
            group_reps.append(angle)
            group_counts.append(0)
            group = len(group_reps) - 1
        radius_by_label[label] = 1.0 - 0.18 * group_counts[group]
        group_counts[group] += 1

    for label, angle in bearings_by_label.items():
        r = radius_by_label[label]
        x, y = bearing_xy(angle)
        x, y = x * r, y * r
        color = colors_by_label[label]
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        if show_labels:
            sector_id = int(((angle - boundary) % 360) * n_sectors // 360)
            ax.annotate(f"{label}\n(sector {sector_id})", (x * arrow_scale, y * arrow_scale), color=color, ha="center", fontsize=9, fontweight="bold")
        else:
            ax.scatter([x], [y], color=color, s=60, zorder=4, edgecolor="white")

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    y0 = -2.0
    if legend:
        legend_source = legend_colors if legend_colors is not None else colors_by_label
        handles = [plt.Line2D([0], [0], color=color, lw=2.5, label=label) for label, color in legend_source.items()]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, (y0 + 1.85) / 1.8), ncol=min(len(handles), 4), frameon=False, fontsize=8)
        y0 -= 0.2 * -(-len(handles) // 4)
    if direction_ids:
        text = "  ".join(f"{label}: {did}" for label, did in direction_ids.items())
        ax.text(0, y0, text, ha="center", va="top", fontsize=8.5)


def plot_split_line_network(ax, stop_coords, branches, split_by_stop, title, line_half_len=0.006):
    """The network (faint), with a dashed local direction-split line
    drawn through every stop that has one. `split_by_stop`: `{stop_id:
    split_angle_degrees_or_None}`.
    """
    for path, color, _label in branches:
        xs = [stop_coords[s][0] for s in path]
        ys = [stop_coords[s][1] for s in path]
        ax.plot(xs, ys, "-", color=color, linewidth=2.5, alpha=0.5, zorder=1)

    for stop_id, split_angle in split_by_stop.items():
        lon, lat = stop_coords[stop_id]
        ax.scatter([lon], [lat], color=CB_BLACK, s=50, zorder=3)
        ax.annotate(stop_id, (lon, lat), textcoords="offset points", xytext=(0, 9), fontsize=10, fontweight="bold", ha="center")
        if split_angle is None:
            continue
        rad = math.radians(split_angle)
        dx, dy = math.sin(rad) * line_half_len, math.cos(rad) * line_half_len
        ax.plot([lon - dx, lon + dx], [lat - dy, lat + dy], "--", color=CB_BLACK, linewidth=1.3, alpha=0.85, zorder=2)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")


def plot_mean_point_map(ax, stop_coords, branches, hub_id, mean_points, title, lower_labels=frozenset(), show_stop_labels=True):
    """The network (faint) plus, from `hub_id`, a dashed arrow to each
    `(mean_lon, mean_lat)` in `mean_points` -- the average-remaining-point
    each `shape_direction` bearing at the hub actually points to.
    `mean_points`: `[(label, color, (lon, lat)), ...]`.

    `show_stop_labels=False` omits every non-hub stop's `Sx`-style text
    label (the dots themselves still show) -- useful once the
    mean-point arrows/legend already carry enough labeling on their own
    and the stop text would just add clutter.
    """
    for path, color, _label in branches:
        xs = [stop_coords[s][0] for s in path]
        ys = [stop_coords[s][1] for s in path]
        ax.plot(xs, ys, "-", color=color, linewidth=2.5, alpha=0.4, zorder=1)

    seen = set()
    for path, _color, _label in branches:
        for sid in path:
            if sid in seen:
                continue
            seen.add(sid)
            lon, lat = stop_coords[sid]
            ax.scatter([lon], [lat], color=CB_BLACK, s=40, zorder=3)
            if not show_stop_labels:
                continue
            below = sid in lower_labels
            dy = -14 if below else 8
            va = "top" if below else "bottom"
            ax.annotate(sid, (lon, lat), textcoords="offset points", xytext=(0, dy), fontsize=9, ha="center", va=va)

    hub_lon, hub_lat = stop_coords[hub_id]
    extent_lons = [lon for lon, _ in stop_coords.values()]
    extent_lats = [lat for _, lat in stop_coords.values()]
    for label, color, (mean_lon, mean_lat) in mean_points:
        ax.annotate("", xy=(mean_lon, mean_lat), xytext=(hub_lon, hub_lat), arrowprops=dict(arrowstyle="->", color=color, lw=2, linestyle="dashed"))
        ax.scatter([mean_lon], [mean_lat], marker="X", s=110, color=color, zorder=4, edgecolor="white", label=label)
        extent_lons.append(mean_lon)
        extent_lats.append(mean_lat)

    ax.scatter([hub_lon], [hub_lat], color=CB_VERMILLION, s=90, zorder=5)
    ax.annotate(hub_id, (hub_lon, hub_lat), textcoords="offset points", xytext=(-10, 8), fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    # Stop/hub labels are drawn via `ax.annotate(..., textcoords="offset
    # points")`, which (like arrows) isn't included in matplotlib's
    # autoscale -- without an explicit margin, a stop near the data's own
    # top/bottom edge gets its label silently clipped at the axes border.
    margin_lon = (max(extent_lons) - min(extent_lons)) * 0.12 or 0.001
    margin_lat = (max(extent_lats) - min(extent_lats)) * 0.12 or 0.001
    ax.set_xlim(min(extent_lons) - margin_lon, max(extent_lons) + margin_lon)
    ax.set_ylim(min(extent_lats) - margin_lat, max(extent_lats) + margin_lat)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)


def plot_departure_timeline(ax, groups, title):
    """A compact horizontal timeline: one row of tick marks per `(label,
    color, departure_seconds, y_lo, y_hi)` in `groups`.
    """
    y_labels, y_positions = [], []
    for label, color, departures, y_lo, y_hi in groups:
        for t in departures:
            ax.axvline(t / 60, color=color, linewidth=2, ymin=y_lo, ymax=y_hi)
        y_positions.append((y_lo + y_hi) / 2)
        y_labels.append(label)
    ax.set_yticks(y_positions, y_labels)
    ax.set_xlabel("minutes into the window")
    ax.set_title(title)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
