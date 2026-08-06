"""Tests for `MobilityDatabaseDownloader.download_historic_stack`.

Exercises the full historic pipeline end to end -- listing dataset
versions, downloading each one's zip, trimming calendars, stitching via
`historic_stack`, and writing the final
`mobility_database_{feed_id}_{start}_{end}.zip` -- against synthetic GTFS
feeds, with only the HTTP layer mocked (no real Mobility Database access,
matching the mocking pattern in `test_mobility_database.py`).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pyGTFSHandler.downloaders.mobility_database import MobilityDatabaseDownloader

from ..gtfs_builder import minimal_agency, write_gtfs


def _patch_token(mock_request_json, expires_in=3600):
    mock_request_json.return_value = {"access_token": "tok", "expires_in": expires_in}


def make_client(**kwargs):
    with patch("pyGTFSHandler.downloaders.mobility_database.request_json") as mock_rj:
        _patch_token(mock_rj)
        client = MobilityDatabaseDownloader(api_key="refresh-token", **kwargs)
    return client


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


def _fake_get(url_to_zip_bytes):
    def _get(url, stream=True, timeout=60):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        content = url_to_zip_bytes[url]
        response.iter_content = MagicMock(return_value=[content])
        return response

    return _get


def test_download_historic_stack_merges_two_dataset_versions(tmp_path):
    dir_a = _publication_dir(
        tmp_path, "pub_a", start_date="20240101", end_date="20241231",
        service_id="SVC_A", trip_id="TRIP_A",
    )
    dir_b = _publication_dir(
        tmp_path, "pub_b", start_date="20240201", end_date="20241231",
        service_id="SVC_B", trip_id="TRIP_B",
    )

    history = [
        {
            "id": "mdb-1-v1",
            "service_date_range_start": "2024-01-01T00:00:00Z",
            "service_date_range_end": "2024-01-31T23:59:59Z",
            "hosted_url": "http://fake/v1.zip",
        },
        {
            "id": "mdb-1-v2",
            "service_date_range_start": "2024-02-01T00:00:00Z",
            "service_date_range_end": "2024-12-31T23:59:59Z",
            "hosted_url": "http://fake/v2.zip",
        },
    ]
    url_to_bytes = {
        "http://fake/v1.zip": _zip_bytes(dir_a),
        "http://fake/v2.zip": _zip_bytes(dir_b),
    }

    client = make_client()
    output_path = tmp_path / "output"

    with patch.object(client, "_authorized_request", return_value=history):
        with patch(
            "pyGTFSHandler.downloaders.mobility_database.requests.get",
            side_effect=_fake_get(url_to_bytes),
        ):
            zip_path = client.download_historic_stack(
                output_path=str(output_path),
                feed_id="mdb-1",
                start_date="2024-01-01",
                end_date="2024-06-30",
            )

    assert zip_path is not None
    assert Path(zip_path).name == "mobility_database_mdb-1_20240101_20240630.zip"
    assert Path(zip_path).is_file()

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("trips.txt") as f:
            trips = pl.read_csv(f.read())
    assert trips.height == 2
    trip_ids = set(trips["trip_id"].to_list())
    assert any("20240101" in t for t in trip_ids)
    assert any("20240201" in t for t in trip_ids)

    # Intermediate per-version folders should be cleaned up.
    assert not (output_path / "mdb-1_start_date_20240101").exists()
    assert not (output_path / "mdb-1").exists()


def test_download_historic_stack_no_matching_versions_returns_none(tmp_path):
    client = make_client()
    output_path = tmp_path / "output"

    with patch.object(client, "_authorized_request", return_value=[]):
        result = client.download_historic_stack(
            output_path=str(output_path),
            feed_id="mdb-empty",
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

    assert result is None


def test_download_historic_stack_skips_when_final_zip_exists(tmp_path):
    client = make_client()
    output_path = tmp_path / "output"
    output_path.mkdir()
    existing_zip = output_path / "mobility_database_mdb-1_20240101_20240630.zip"
    existing_zip.write_bytes(b"stale-zip-contents")

    with patch.object(client, "_authorized_request") as mock_history:
        zip_path = client.download_historic_stack(
            output_path=str(output_path),
            feed_id="mdb-1",
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

    mock_history.assert_not_called()
    assert zip_path == str(existing_zip)
