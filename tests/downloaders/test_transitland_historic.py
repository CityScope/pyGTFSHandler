"""Tests for `TransitLandDownloader.download_historic_stack`.

Same end-to-end shape as `test_mobility_database_historic.py`: synthetic
GTFS feeds stand in for real feed_version downloads, only the HTTP layer
is mocked, and the full listing -> download -> trim -> stitch -> zip
pipeline runs for real.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pyGTFSHandler.downloaders.transitland import TransitLandDownloader

from ..gtfs_builder import minimal_agency, write_gtfs


def make_client(**kwargs):
    return TransitLandDownloader(api_key="tl-key", **kwargs)


def _zip_bytes(directory: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in directory.iterdir():
            zf.write(f, arcname=f.name)
    return buf.getvalue()


def _publication_dir(tmp_path, name, *, start_date, end_date, service_id, trip_id):
    directory = tmp_path / name
    write_gtfs(
        directory,
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {
                    "service_id": service_id,
                    "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                    "friday": 1, "saturday": 0, "sunday": 0,
                    "start_date": start_date, "end_date": end_date,
                }
            ],
            "routes.txt": [
                {"route_id": "R1", "route_short_name": "R1", "route_long_name": "Line", "route_type": 3}
            ],
            "stops.txt": [
                {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 40.0, "stop_lon": -3.7},
                {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 40.01, "stop_lon": -3.69},
            ],
            "trips.txt": [{"route_id": "R1", "service_id": service_id, "trip_id": trip_id}],
            "stop_times.txt": [
                {"trip_id": trip_id, "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": trip_id, "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )
    return directory


def _fake_get(url_to_zip_bytes, status_code=200):
    def _get(url, params=None, stream=True, timeout=60):
        response = MagicMock()
        response.status_code = status_code
        response.raise_for_status = MagicMock()
        content = url_to_zip_bytes.get(url, b"")
        response.iter_content = MagicMock(return_value=[content])
        return response

    return _get


def test_download_historic_stack_merges_two_feed_versions(tmp_path):
    dir_a = _publication_dir(
        tmp_path, "pub_a", start_date="20240101", end_date="20241231",
        service_id="SVC_A", trip_id="TRIP_A",
    )
    dir_b = _publication_dir(
        tmp_path, "pub_b", start_date="20240201", end_date="20241231",
        service_id="SVC_B", trip_id="TRIP_B",
    )

    history = {
        "feed_versions": [
            {
                "sha1": "v1sha",
                "earliest_calendar_date": "2024-01-01",
                "latest_calendar_date": "2024-01-31",
            },
            {
                "sha1": "v2sha",
                "earliest_calendar_date": "2024-02-01",
                "latest_calendar_date": "2024-12-31",
            },
        ]
    }
    url_to_bytes = {
        f"{TransitLandDownloader.BASE_URL}/feed_versions/v1sha/download": _zip_bytes(dir_a),
        f"{TransitLandDownloader.BASE_URL}/feed_versions/v2sha/download": _zip_bytes(dir_b),
    }

    client = make_client()
    output_path = tmp_path / "output"

    with patch.object(client, "_get", return_value=history):
        with patch(
            "pyGTFSHandler.downloaders.transitland.requests.get",
            side_effect=_fake_get(url_to_bytes),
        ):
            zip_path = client.download_historic_stack(
                output_path=str(output_path),
                feed_key="f-onestop",
                start_date="2024-01-01",
                end_date="2024-06-30",
            )

    assert zip_path is not None
    assert Path(zip_path).name == "transitland_f-onestop_20240101_20240630.zip"
    assert Path(zip_path).is_file()

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("trips.txt") as f:
            trips = pl.read_csv(f.read())
    assert trips.height == 2
    trip_ids = set(trips["trip_id"].to_list())
    assert any("20240101" in t for t in trip_ids)
    assert any("20240201" in t for t in trip_ids)


def test_download_historic_stack_raises_permission_error_on_gated_download(tmp_path):
    history = {
        "feed_versions": [
            {
                "sha1": "old_sha",
                "earliest_calendar_date": "2024-01-01",
                "latest_calendar_date": "2024-12-31",
            }
        ]
    }
    client = make_client()
    output_path = tmp_path / "output"

    with patch.object(client, "_get", return_value=history):
        with patch(
            "pyGTFSHandler.downloaders.transitland.requests.get",
            side_effect=_fake_get({}, status_code=402),
        ):
            with pytest.raises(PermissionError):
                client.download_historic_stack(
                    output_path=str(output_path),
                    feed_key="f-gated",
                    start_date="2024-01-01",
                    end_date="2024-06-30",
                )


def test_download_historic_stack_no_matching_versions_returns_none(tmp_path):
    client = make_client()
    output_path = tmp_path / "output"

    with patch.object(client, "_get", return_value={"feed_versions": []}):
        result = client.download_historic_stack(
            output_path=str(output_path),
            feed_key="f-empty",
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

    assert result is None
