# -*- coding: utf-8 -*-
"""Marking Bus Rapid Transit routes in downloaded NTD GTFS feeds.

NTD lists "Bus Rapid Transit" as its own mode, but GTFS itself has no
route_type value for BRT anywhere in its spec -- not in the basic 0-7
enum, nor in the extended 100-1799 hierarchy (verified 2026-09-29 against
Google's extended-route-types reference; the closest real code, 702
"Express Bus Service", is a different concept and not used as a stand-in
here). Agencies publish these routes as plain route_type=3 (Bus).

So when NTD flags a feed as BRT, this module overwrites the relevant
routes' `route_type` in-place, in OUR downloaded copy only (never the
source agency's own published feed), to `BRT_ROUTE_TYPE` -- a value chosen
to collide with nothing real (unused by both the basic enum and the
extended hierarchy, which starts at 100). Two cases:

1. **BRT is the feed's only bus-family mode, and route_type is unique**:
   if NTD lists no OTHER bus-family mode ("Bus", "Commuter Bus",
   "Trolleybus") against this same feed URL, and every route in the feed
   already shares the same route_type, no route-level knowledge is
   needed -- the whole feed is BRT, so every row is overwritten.
2. **BRT shares the feed with plain bus service, or route_type is mixed**:
   many real agencies (e.g. LA Metro) run their BRT line(s) and their
   ordinary bus routes out of the SAME feed at the SAME route_type=3 --
   a uniform route_type alone does NOT mean "the whole feed is BRT" here,
   so this case is never auto-overwritten wholesale. `KNOWN_BRT_ROUTES`
   is a small, verified (not memory-guessed) seed table of specific
   agencies' real route_short_name/route_id values for exactly this
   case; anything not in that table is left alone and logged for a
   manual, route-by-route look before the next run.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Union

import polars as pl

logger = logging.getLogger(__name__)

#: NTD modes that, like BRT, ordinarily encode as route_type=3 (Bus) --
#: if any of these coexist with "Bus Rapid Transit" against the same feed
#: URL, the feed mixes BRT with plain bus service and case 1 (blanket
#: unique-route_type overwrite) is unsafe.
_OTHER_BUS_FAMILY_MODES = {"Bus", "Commuter Bus", "Trolleybus"}

#: route_type marker written for Bus Rapid Transit in OUR downloaded feed
#: copies. Per project owner's explicit decision (2026-09-29): reuses
#: GTFS's real extended code 702 ("Express Bus Service"), the closest
#: existing spec value to BRT, rather than a value with no meaning to any
#: other GTFS-aware tool. Caveat carried over from that decision: 702's
#: official spec meaning is Express Bus Service, a related but distinct
#: concept (limited-stop bus service, with or without BRT infrastructure)
#: -- so a route rewritten here to 702 is a deliberate repurposing, not a
#: literal reading of the GTFS spec, and any downstream consumer of these
#: files that assumes 702 means "express bus" rather than "our BRT" will
#: be misled. Scoped to this project's own downloaded copies only, never
#: the source agency's published feed.
BRT_ROUTE_TYPE = 702

#: Verified (not memory-guessed) route_short_name/route_id values for
#: agencies whose BRT lines share a feed with non-BRT service at a
#: non-unique route_type, keyed by NTD agency id. Each value is the set
#: of `route_short_name` OR `route_id` strings (matched against both
#: columns) that are that agency's actual BRT line(s) in their published
#: GTFS. Extend this table only with entries checked against a real
#: source (the feed itself, transit.land, or a citable secondary source
#: quoting the feed) -- see `brt.py`'s module docstring.
#:
#: Seeded 2026-09-29 from a verified research pass (transit.land route
#: pages / agency route materials, not memory alone), NTD ids
#: cross-checked directly against a live NTD query (`ntd_id` field), which
#: corrected two of the research pass's own ids:
#: - GCRTA (Cleveland), NTD 50015: HealthLine, route_short_name "HL"
#:   (https://www.transit.land/routes/r-dpmu-hl)
#: - ABQ RIDE (Albuquerque), NTD 60019 (research pass had "6019" --
#:   wrong; corrected against a live NTD query): ART Red Line,
#:   route_short_name "766"
#: - LA Metro, NTD 90154 (research pass flagged this id as unconfirmed --
#:   now confirmed against a live NTD query): G Line ("901") and J Line
#:   ("910"/"950")
#: Still unverified (real BRT lines, but no citable GTFS field value
#: found via web search -- needs a direct feed pull, not search snippets):
#: Pittsburgh PRT East/West Busway, Eugene LTD EmX, Kansas City RideKC
#: MAX, Richmond GRTC Pulse, Minneapolis METRO A/C/D.
KNOWN_BRT_ROUTES: Dict[str, Set[str]] = {
    "50015": {"HL"},
    "60019": {"766"},
    "90154": {"901", "910", "950"},
}


def _load_routes(routes_path: Path) -> pl.DataFrame:
    return pl.read_csv(routes_path, infer_schema_length=None)


def resolve_and_apply_brt(
    gtfs_dir: Union[str, Path], ntd_id: str, feed_ntd_modes: Set[str]
) -> Dict[str, object]:
    """Overwrite BRT routes' `route_type` in a downloaded feed, in place.

    Only called for feeds whose NTD rows include the "Bus Rapid Transit"
    mode (the caller is responsible for that check, since it needs the
    NTD rows, not just the feed on disk).

    Args:
        gtfs_dir: Path to the extracted GTFS feed directory (its
            `routes.txt` is rewritten in place if any route is matched).
        ntd_id: The feed's NTD agency id, used to look up
            `KNOWN_BRT_ROUTES` for the mixed-mode/mixed-route_type case.
        feed_ntd_modes: Every NTD `mode_name` associated with this same
            feed URL (not just "Bus Rapid Transit") -- used to detect
            when BRT shares the feed with ordinary bus service, in which
            case a uniform route_type does NOT mean "the whole feed is
            BRT" (see module docstring, case 2).

    Returns:
        A dict with `"overwritten"` (int, rows changed) and `"matched_by"`
        (`"unique_route_type"`, `"known_routes"`, or `"none"` if nothing
        could be resolved and this feed needs manual review).
    """
    routes_path = Path(gtfs_dir) / "routes.txt"
    if not routes_path.is_file():
        logger.warning(f"No routes.txt in '{gtfs_dir}'; cannot apply BRT override.")
        return {"overwritten": 0, "matched_by": "none"}

    routes = _load_routes(routes_path)
    unique_types = routes["route_type"].unique().to_list()
    mixes_with_plain_bus = bool(feed_ntd_modes & _OTHER_BUS_FAMILY_MODES)

    if len(unique_types) == 1 and not mixes_with_plain_bus:
        routes = routes.with_columns(pl.lit(BRT_ROUTE_TYPE).alias("route_type"))
        routes.write_csv(routes_path)
        logger.info(
            f"NTD BRT feed '{ntd_id}': single route_type ({unique_types[0]}), BRT is the "
            f"feed's only bus-family mode -- overwrote all {routes.height} routes to "
            f"BRT_ROUTE_TYPE={BRT_ROUTE_TYPE}."
        )
        return {"overwritten": routes.height, "matched_by": "unique_route_type"}

    known = KNOWN_BRT_ROUTES.get(str(ntd_id))
    if known:
        present_ids = set(routes["route_short_name"].cast(pl.Utf8)) | set(
            routes["route_id"].cast(pl.Utf8)
        )
        stale = known - present_ids
        if stale:
            # Automatic integrity check, every download: a KNOWN_BRT_ROUTES
            # identifier that verified fine against transit.land on the day
            # it was seeded can still go stale later if the agency renames
            # or renumbers the line -- surface that immediately rather than
            # silently under-matching.
            logger.warning(
                f"NTD BRT feed '{ntd_id}': KNOWN_BRT_ROUTES has {sorted(stale)} but "
                f"none of these appear in this download's routes.txt (route_short_name/"
                "route_id) -- the agency may have renamed/renumbered this line since "
                "KNOWN_BRT_ROUTES was seeded. Verify and update the table."
            )

        mask = pl.col("route_short_name").cast(pl.Utf8).is_in(list(known)) | pl.col(
            "route_id"
        ).cast(pl.Utf8).is_in(list(known))
        n_matched = routes.filter(mask).height
        if n_matched:
            routes = routes.with_columns(
                pl.when(mask).then(pl.lit(BRT_ROUTE_TYPE)).otherwise(pl.col("route_type")).alias(
                    "route_type"
                )
            )
            routes.write_csv(routes_path)
            logger.info(
                f"NTD BRT feed '{ntd_id}': mixed route_type ({unique_types}) -- overwrote "
                f"{n_matched} known BRT route(s) to BRT_ROUTE_TYPE={BRT_ROUTE_TYPE} via "
                "KNOWN_BRT_ROUTES."
            )
            return {"overwritten": n_matched, "matched_by": "known_routes"}

    logger.warning(
        f"NTD BRT feed '{ntd_id}': route_type={unique_types}, mixes_with_plain_bus="
        f"{mixes_with_plain_bus}, no KNOWN_BRT_ROUTES entry -- needs manual "
        "route-by-route review. Left unchanged."
    )
    return {"overwritten": 0, "matched_by": "none"}


def apply_brt_overrides_for_feed(
    gtfs_dir: Union[str, Path], ntd_rows: List[Dict[str, object]]
) -> Dict[str, object]:
    """Apply the BRT override to a downloaded feed if any of its NTD rows say BRT.

    Args:
        gtfs_dir: Path to the extracted GTFS feed directory.
        ntd_rows: EVERY NTD row (raw dicts, as in
            `GTFSFeedMetadata.raw["ntd"]`) whose `download_url` points at
            this same `gtfs_dir` -- not just its BRT row(s), since telling
            whether BRT shares the feed with plain bus service requires
            seeing every mode NTD lists against this feed URL.

    Returns:
        The result of `resolve_and_apply_brt`, or
        `{"overwritten": 0, "matched_by": "not_brt"}` if none of
        `ntd_rows` is BRT.
    """
    feed_ntd_modes = {r.get("mode_name") for r in ntd_rows}
    if "Bus Rapid Transit" not in feed_ntd_modes:
        return {"overwritten": 0, "matched_by": "not_brt"}

    brt_rows = [r for r in ntd_rows if r.get("mode_name") == "Bus Rapid Transit"]
    ntd_id = str(brt_rows[0].get("ntd_id", ""))
    return resolve_and_apply_brt(gtfs_dir, ntd_id, feed_ntd_modes)
