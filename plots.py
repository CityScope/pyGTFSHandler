# -*- coding: utf-8 -*-
"""Generates the illustrative JPGs embedded in the Methodology docs pages.

Not part of the published site or the package -- a standalone script you run
locally to (re)generate `docs/assets/img/*.jpg`. Reuses the same plotting
helpers (`pyGTFSHandler.plots.example_notebooks`) and synthetic Y-shaped test
network (`tests.example_gtfs`) that `examples/direction_and_headway_methodology.ipynb`
uses, plus two new small figures (frequency expansion, calendar resolution)
built directly from `Feed`/`Calendar` for the other methodology pages.

Usage:
    uv sync --extra plot --extra docs
    uv run python docs/plots.py
"""

import sys
import tempfile
import warnings
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pyGTFSHandler import Feed
from pyGTFSHandler.models.calendar import Calendar
from pyGTFSHandler.models.shapes import _reconcile_fwd_bwd, _split_angle
from pyGTFSHandler.plots.example_notebooks import (
    compute_headway,
    plot_bearing_correction,
    plot_departure_timeline,
    plot_headway_bar,
    plot_network,
    plot_widest_gap_circle,
)
from pyGTFSHandler.utils.colors import CB_BLACK, CB_BLUE, CB_GREEN, CB_ORANGE, CB_VERMILLION
from tests.example_gtfs import Y_STOP_COORDS as STOP_COORDS
from tests.example_gtfs import build_y_shape_variants

warnings.filterwarnings("ignore", category=DeprecationWarning)

OUT_DIR = Path(__file__).resolve().parent / "assets" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRUNK = ["S1", "S2", "S3"]
BRANCH1 = ["S3", "S4", "S5"]
BRANCH2 = ["S3", "S6", "S7"]
LOWER = {"S6", "S7"}


def savefig(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def fig_network_and_headway_grouping(feeds: dict) -> None:
    """For methodology/direction-headway.md: the Y-network, and why
    grouping by `route_id` gives a different (wrong) trunk headway than
    grouping by geometric `shape_direction`."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    plot_network(
        ax,
        STOP_COORDS,
        [(TRUNK, CB_BLACK, "shared trunk"), (BRANCH1, CB_BLUE, "shape 1 (-> S5)"), (BRANCH2, CB_ORANGE, "shape 2 (-> S7)")],
        title="The Y-shaped network -- one trunk, two branches",
        lower_labels=LOWER,
    )
    ax.set_ylim(40.378, 40.423)
    savefig(fig, "network_y_shape.jpg")

    start_sec, end_sec = 7 * 3600, 9 * 3600
    # Both branches run every 10 min on the trunk, offset by 5 min (as
    # `tests.example_gtfs.build_y_shape`'s default `offset2=300` does) --
    # together the trunk actually sees a departure every 5 min.
    t1 = np.arange(0, end_sec - start_sec, 600)
    t2 = np.arange(300, end_sec - start_sec, 600)
    merged = np.sort(np.concatenate([t1, t2]))

    # diff_routes feed: grouping by `route_id` only ever sees one branch's
    # own departures on the shared trunk edge -- 10 min, understating the
    # true 5 min trunk frequency.
    route_id_headway_diff = compute_headway(t1, 0, end_sec - start_sec)
    # same_route feed: both branches share one `route_id`, so grouping by
    # `route_id` happens to merge them correctly here -- 5 min.
    route_id_headway_same = compute_headway(merged, 0, end_sec - start_sec)
    # Grouping by geometric `shape_direction` always merges same-direction
    # traffic on an edge, regardless of route modeling -- 5 min either way.
    shape_direction_headway = compute_headway(merged, 0, end_sec - start_sec)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plot_headway_bar(
        ax,
        labels=["route_id\n(diff. route_id)", "route_id\n(same route_id)", "shape_direction\n(either)"],
        values=[route_id_headway_diff / 60, route_id_headway_same / 60, shape_direction_headway / 60],
        colors=[CB_VERMILLION, CB_ORANGE, CB_BLUE],
        title="Same network, same schedule -- grouping choice changes the answer",
        ylabel="reported trunk headway (S2), minutes",
    )
    savefig(fig, "headway_grouping_comparison.jpg")


def fig_bearing_correction_and_split(feeds: dict) -> None:
    """For methodology/direction-headway.md: reconciling forward/backward
    shape bearings at a stop, then splitting them into direction bins via
    the widest-gap rule."""
    feed = Feed(str(feeds["diff_routes"]))
    bearings = (
        feed.shapes.stop_shapes.filter(pl.col("stop_id") == "S3_file_0")
        .select(["shape_id", "shape_direction", "shape_direction_backwards"])
        .collect()
    )
    raw = {row["shape_id"]: (row["shape_direction"], row["shape_direction_backwards"]) for row in bearings.iter_rows(named=True)}
    shape_colors = {sid: (CB_BLUE if i == 0 else CB_ORANGE) for i, sid in enumerate(raw)}

    fig, ax = plt.subplots(figsize=(6, 6.5), subplot_kw={"aspect": "equal"})
    plot_bearing_correction(fig, ax, raw, shape_colors, _reconcile_fwd_bwd)
    ax.set_title("Forward/backward bearings at S3, before and after 180 deg correction", fontsize=12)
    savefig(fig, "bearing_correction.jpg")

    corrected = {sid: _reconcile_fwd_bwd(fwd, bwd) for sid, (fwd, bwd) in raw.items()}
    split = _split_angle(corrected)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"aspect": "equal"})
    plot_widest_gap_circle(ax, corrected, split, shape_colors)
    ax.set_title("Widest-gap split into direction bins", fontsize=12)
    savefig(fig, "widest_gap_split.jpg")


def fig_combined_departures(feeds: dict) -> None:
    """For methodology/direction-headway.md: two independently-scheduled
    services sharing a trunk edge don't just "interleave" into a clean
    combined headway."""
    start_sec, end_sec = 0, 4 * 3600
    t1 = np.arange(0, end_sec, 600)          # every 10 min
    t2 = np.arange(300, end_sec, 1200)        # every 20 min, offset 5 min

    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    plot_departure_timeline(
        ax,
        groups=[("shape 1\n(10 min)", CB_BLUE, t1, 0.55, 1.0), ("shape 2\n(20 min)", CB_ORANGE, t2, 0.0, 0.45)],
        title="Combined trunk departures: not simply interleaved",
    )
    ax.set_xlim(0, end_sec / 60)  # plot_departure_timeline plots t/60 (minutes)
    savefig(fig, "combined_departures_timeline.jpg")


def fig_frequency_expansion() -> None:
    """For methodology/frequencies.md: `frequencies.txt`'s headway-based
    trips (T1: every 10 min, T3: every 15 min, both 07:00-09:00) expanded
    into concrete departure times."""
    gtfs_dir = REPO_ROOT / "examples" / "test_files" / "shape_test"
    feed = Feed(str(gtfs_dir))

    freq_lf = pl.read_csv(gtfs_dir / "frequencies.txt")
    # NB: `Feed.filter`'s `frequencies` flag is the *keep-as-template* switch,
    # not "expand" -- frequencies=False is what calls
    # `_frequencies_to_stop_times` and expands into concrete departures.
    expanded = (
        feed.filter(frequencies=False)
        .filter(pl.col("stop_sequence") == pl.col("stop_sequence").min().over("trip_id"))
        .select(["trip_id", "departure_time"])
        .sort(["trip_id", "departure_time"])
        .collect()
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 3.2), sharex=True)
    colors = {"T1": CB_BLUE, "T2": CB_ORANGE, "T3": CB_GREEN, "T4": CB_VERMILLION}

    # Top: the frequencies.txt definition, drawn as a shaded window + headway label.
    ax = axes[0]
    for i, row in enumerate(freq_lf.iter_rows(named=True)):
        trip_id = row["trip_id"]
        start_sec = _hms_to_sec(row["start_time"])
        end_sec = _hms_to_sec(row["end_time"])
        y = i
        ax.plot([start_sec, end_sec], [y, y], color=colors.get(trip_id, CB_BLACK), linewidth=6, solid_capstyle="butt")
        ax.text(end_sec + 120, y, f"{trip_id}: every {row['headway_secs']}s", va="center", fontsize=9)
    ax.set_yticks([])
    ax.set_title("frequencies.txt: start_time / end_time / headway_secs per trip_id", fontsize=10, loc="left")
    ax.set_ylim(-1, len(freq_lf))

    # Bottom: the expanded concrete departures pyGTFSHandler produces.
    ax = axes[1]
    root_trip_of = lambda t: t.split("_")[0] if "_" in t else t
    ordered_roots = list(dict.fromkeys(freq_lf["trip_id"].to_list()))
    for i, root in enumerate(ordered_roots):
        secs = [
            _hms_to_sec(r["departure_time"]) if isinstance(r["departure_time"], str) else r["departure_time"]
            for r in expanded.iter_rows(named=True)
            if root_trip_of(r["trip_id"]) == root
        ]
        ax.scatter(secs, [i] * len(secs), color=colors.get(root, CB_BLACK), s=18, zorder=3)
    ax.set_yticks(range(len(ordered_roots)))
    ax.set_yticklabels(ordered_roots)
    ax.set_title("Expanded into concrete departures (one dot per generated trip)", fontsize=10, loc="left")
    ax.set_xlabel("time of day")
    ax.set_xticks([25200, 27000, 28800, 30600, 32400])
    ax.set_xticklabels(["07:00", "07:30", "08:00", "08:30", "09:00"])
    for ax_ in axes:
        ax_.spines[["top", "right", "left"]].set_visible(False)
        ax_.set_xlim(25200, 32400)
    plt.tight_layout()
    savefig(fig, "frequency_expansion.jpg")


def _hms_to_sec(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def fig_calendar_resolution() -> None:
    """For methodology/calendar.md: a weekly `calendar.txt` pattern with a
    `calendar_dates.txt` addition (extra service on a Sunday) and removal
    (a Monday holiday) resolved into the actual set of active dates."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        start = date(2024, 5, 6)  # a Monday
        end = start + timedelta(days=13)

        (tmp / "calendar.txt").write_text(
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
            f"WEEKDAY,1,1,1,1,1,0,0,{start:%Y%m%d},{end:%Y%m%d}\n"
        )
        # +1 = added (exception_type 1), 2 = removed (exception_type 2)
        holiday_monday = start + timedelta(days=7)  # 2nd Monday: removed
        extra_sunday = start + timedelta(days=6)    # 1st Sunday: added
        (tmp / "calendar_dates.txt").write_text(
            "service_id,date,exception_type\n"
            f"WEEKDAY,{holiday_monday:%Y%m%d},2\n"
            f"WEEKDAY,{extra_sunday:%Y%m%d},1\n"
        )

        cal = Calendar()
        cal.load([tmp], start_date=start, end_date=end)

        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        active = [bool(cal.get_services_in_date(d)) for d in days]

    fig, ax = plt.subplots(figsize=(9, 1.6))
    for i, (d, is_active) in enumerate(zip(days, active)):
        color = CB_BLUE if is_active else "#d9d9d9"
        ax.bar(i, 1, color=color, edgecolor="white", width=0.9)
        label = d.strftime("%a\n%d")
        ax.text(i, -0.35, label, ha="center", va="top", fontsize=8)
    ax.annotate(
        "calendar_dates.txt:\n+1 (added)", xy=(6, 1.05), xytext=(4.3, 2.0),
        ha="center", fontsize=8, color=CB_GREEN,
        arrowprops=dict(arrowstyle="->", color=CB_GREEN),
    )
    ax.annotate(
        "calendar_dates.txt:\nremoved (holiday)", xy=(7, 0.5), xytext=(8.7, 2.0),
        ha="center", fontsize=8, color=CB_VERMILLION,
        arrowprops=dict(arrowstyle="->", color=CB_VERMILLION),
    )
    ax.set_xlim(-0.6, len(days) - 0.4)
    ax.set_ylim(-0.6, 2.7)
    ax.axis("off")
    ax.set_title("calendar.txt (Mon-Fri) resolved with two calendar_dates.txt exceptions", fontsize=10)
    savefig(fig, "calendar_resolution.jpg")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        feeds = build_y_shape_variants(Path(tmp))
        fig_network_and_headway_grouping(feeds)
        fig_bearing_correction_and_split(feeds)
        fig_combined_departures(feeds)
    fig_frequency_expansion()
    fig_calendar_resolution()


if __name__ == "__main__":
    main()
