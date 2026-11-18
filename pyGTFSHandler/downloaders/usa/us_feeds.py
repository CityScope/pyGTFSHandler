# -*- coding: utf-8 -*-
"""Download every US GTFS feed of a study from one of two lists.

``source="official"``
    Only the FTA National Transit Database weblinks (`NTDDownloader`): feeds whose link answers with a static GTFS zip.
``source="revised"``
    The official list plus a revised list given by the caller (``revised_csv``; the package ships no list -- the US study keeps its own
    in ``TransitLOSStudies/usa_study/feed_lists/``): systems whose NTD link is dead, blocked, partial or points to the wrong agency
    (columns ``state, agency, ref, note``). ``ref`` is one of: a direct zip URL; a catalog id ``mdb-N`` / ``tld-N`` / ``ntd-N``
    (downloaded from the Mobility Database archive ``https://files.mobilitydatabase.org/<id>/latest.zip``); a Transitland feed id
    ``f-...`` (archived latest version through the Transitland API, key from ``TRANSITLAND_API_KEY`` or ``api_keys.json``); or
    ``search:<name>`` (first Transitland feed whose operator matches the name). In revised mode the official NTD feeds listed in
    ``replace_json`` (``{ntd_id: agency}``, partial feeds: one route or mode of a large agency) are NOT downloaded; the revised rows replace them.

Each revised feed is unpacked to ``<folder>/revised_<id or slug>``; a failed one is reported and skipped, never fatal.
"""
from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

TRANSITLAND_BASE = "https://transit.land/api/v2/rest"
ARCHIVE_URL = "https://files.mobilitydatabase.org/{id}/latest.zip"
_UA = {"User-Agent": "Mozilla/5.0 pyGTFSHandler"}


def load_revised_feeds(revised_csv, states: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
    """Rows of a revised feed list; with `states` (abbreviations), the rows of those states plus the ones without a state.

    Args:
        revised_csv: CSV with columns ``state, agency, ref, note``.
        states: State abbreviations; ``None`` = all rows.

    Returns:
        The CSV rows as dicts.
    """
    want = {s.upper() for s in states} if states else None
    with open(revised_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if want is None or not r["state"] or r["state"].upper() in want]


def replace_ntd_ids(replace_json) -> Dict[str, str]:
    """NTD ids whose official weblink is a partial feed, with the agency name: skipped in revised mode.

    Args:
        replace_json: JSON file ``{ntd_id: agency name}``; ``None`` = none skipped.

    Returns:
        ``{ntd_id: agency name}``.
    """
    import json

    return json.loads(Path(replace_json).read_text(encoding="utf-8")) if replace_json else {}


def _transitland_key() -> Optional[str]:
    from ..utils.config import get_api_key

    return get_api_key("transitland", None, "TRANSITLAND_API_KEY") or None


def _fetch_zip(ref: str, timeout: int) -> bytes:
    """Bytes of the zip for a revised `ref` (see the module docstring for the kinds)."""
    if ref.startswith("http"):
        url, params = ref, None
    elif ref.startswith(("f-", "search:")):
        key = _transitland_key()
        if not key:
            raise RuntimeError("no Transitland API key")
        if ref.startswith("search:"):
            hit = requests.get(f"{TRANSITLAND_BASE}/feeds", params={"apikey": key, "search": ref[7:], "spec": "gtfs"}, timeout=60).json().get("feeds", [])
            if not hit:
                raise RuntimeError("no Transitland feed found")
            ref = hit[0]["onestop_id"]
        url, params = f"{TRANSITLAND_BASE}/feeds/{ref}/download_latest_feed_version", {"apikey": key}
    else:
        url, params = ARCHIVE_URL.format(id=ref), None
    resp = requests.get(url, params=params, timeout=timeout, headers=_UA)
    resp.raise_for_status()
    return resp.content


def _folder_name(row: Dict[str, str]) -> str:
    ref = row["ref"]
    if ref.startswith(("mdb-", "tld-", "ntd-", "f-")):
        tail = ref
    else:
        tail = re.sub(r"[^a-z0-9]+", "_", (row["agency"] + "_" + ref if ref.startswith("search:") else row["agency"]).lower()).strip("_")[:60]
    return f"revised_{tail}"


def _is_static_gtfs(folder: Path) -> bool:
    names = {p.name.lower() for p in folder.rglob("*") if p.is_file()}
    return all(f"{b}.txt" in names for b in ("stops", "trips", "stop_times"))


def download_revised_feed(row: Dict[str, str], download_folder: str, timeout: int = 180) -> Optional[Path]:
    """Download and unpack one revised feed; None (with a warning) when it is unreachable or not a static GTFS zip.

    Args:
        row: One row of the revised list (``agency``, ``ref``, ``state``...).
        download_folder: Folder that receives the feed.
        timeout: Download timeout in seconds.

    Returns:
        The feed folder, or None.
    """
    dest = Path(download_folder) / _folder_name(row)
    if dest.is_dir() and _is_static_gtfs(dest):
        return dest
    try:
        content = _fetch_zip(row["ref"], timeout)
        dest.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(content)).extractall(dest)
    except Exception as exc:  # noqa: BLE001 -- one bad link must not stop the rest
        logger.warning(f"revised feed {row['agency']} ({row['ref']}) failed: {type(exc).__name__}: {str(exc)[:80]}")
        return None
    if not _is_static_gtfs(dest):
        logger.warning(f"revised feed {row['agency']} ({row['ref']}) is not a static GTFS zip")
        return None
    return dest


def download_us_feeds(download_folder: str, source: str = "official", states: Optional[Iterable[str]] = None,
                      revised_csv=None, replace_json=None) -> Dict[str, List[Path]]:
    """Download the US feeds of `states` (all states when omitted) into ``<download_folder>/<STATE>/``.

    `source` is ``"official"`` (NTD list only) or ``"revised"`` (NTD list + the caller's revised list). Rows of the revised list
    without a state go to ``<download_folder>/_national``.

    Args:
        download_folder: Root folder; one sub-folder per state.
        source: ``"official"`` or ``"revised"``.
        states: State abbreviations; ``None`` = every state.
        revised_csv: Revised feed list (``state, agency, ref, note``); required with ``source="revised"``.
        replace_json: ``{ntd_id: agency}`` of the partial NTD feeds the revised rows replace.

    Returns:
        ``{state: [feed folders]}``.

    Raises:
        ValueError: For an unknown ``source``, or ``source="revised"`` without ``revised_csv``.
    """
    if source not in ("official", "revised"):
        raise ValueError("source must be 'official' or 'revised'")
    if source == "revised" and not revised_csv:
        raise ValueError("source='revised' needs revised_csv (the package ships no feed list)")
    from .ntd import NTDDownloader

    ntd = NTDDownloader()
    out: Dict[str, List[Path]] = {}
    state_list = [s.upper() for s in states] if states else sorted({str(f.raw["ntd"].get("state")) for f in ntd.search_feeds() if f.raw["ntd"].get("state")})
    for st in state_list:
        folder = Path(download_folder) / st
        folder.mkdir(parents=True, exist_ok=True)
        feeds = ntd.search_feeds(state=st)
        working, broken = ntd.check_feed_links(feeds)
        if broken:
            logger.info(f"{st}: {len(broken)} NTD link(s) unusable")
        if source == "revised":
            skip = replace_ntd_ids(replace_json)
            partial = [f for f in working if str(f.raw["ntd"].get("ntd_id")) in skip]
            if partial:
                logger.info(f"{st}: {len(partial)} partial NTD feed(s) replaced by revised ones")
            working = [f for f in working if f not in partial]
        ntd.download_feeds(working, str(folder))
        out[st] = [p for p in folder.iterdir() if p.is_dir()]
    if source == "revised":
        for row in load_revised_feeds(revised_csv, state_list if states else None):
            st = row["state"].upper() or "_national"
            folder = Path(download_folder) / st
            folder.mkdir(parents=True, exist_ok=True)
            got = download_revised_feed(row, str(folder))
            if got:
                out.setdefault(st, []).append(got)
    return out
