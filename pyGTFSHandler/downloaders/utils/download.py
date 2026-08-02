# -*- coding: utf-8 -*-
"""Shared "download a list of feeds to disk" loop used by every downloader.

`search_feeds()` on any downloader returns a list of
`downloaders.utils.models.GTFSFeedMetadata`. Once a list of feeds has been
found, downloading them is the same operation regardless of which catalog
they came from: stream each `download_url` to a ZIP file inside
`download_folder`, skip or overwrite feeds that already exist on disk, and
optionally unzip the result. `download_feeds` implements that loop once so
`BaseGTFSDownloader.download_feeds` (and any downloader) can reuse it
instead of re-implementing the same file-handling logic.
"""

import logging
import os
import shutil
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

from ...utils import io
from .models import GTFSFeedMetadata
from .naming import build_feed_filename

logger = logging.getLogger(__name__)


def download_feeds(
    feeds: List[GTFSFeedMetadata],
    download_folder: str,
    overwrite: bool = False,
    unzip: bool = True,
    request_timeout: int = 60,
    headers: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Download a list of GTFS feeds to a folder.

    Streams each feed's `download_url` into `download_folder` as a ZIP
    file. Feeds that already have a matching ZIP or extracted folder on
    disk are skipped unless `overwrite` is set. Feeds with no usable
    `download_url` are skipped with a warning.

    Args:
        feeds: Feeds to download, as returned by a downloader's
            `search_feeds()`.
        download_folder: Directory to store (and, if `unzip`, extract) the
            downloaded feeds into. Created if it doesn't exist.
        overwrite: If True, re-download and replace files that already
            exist on disk.
        unzip: If True, extract each ZIP after downloading and delete the
            ZIP file, returning paths to the extracted folders.
        request_timeout: Timeout in seconds for each download request.
        headers: Optional HTTP headers sent with every download request,
            for sources (e.g. Spain's NAP) whose download links require
            authentication.

    Returns:
        Absolute paths to the downloaded feeds (ZIP files, or extracted
        folders if `unzip` is True), in the same order as `feeds`
        (skipped feeds are simply absent).
    """
    os.makedirs(download_folder, exist_ok=True)
    zip_paths: List[str] = []
    logged_overwrite = False
    logged_skip = False

    for feed in tqdm(feeds, desc="Downloading feeds"):
        if not feed.download_url:
            logger.warning(f"Feed '{feed.id}' has no download URL. Skipping.")
            continue

        filename = build_feed_filename(feed.id, feed.name or "", feed.provider or "")
        if not filename:
            logger.warning(f"Skipping feed with no usable id/name/provider: {feed}")
            continue

        zip_path = os.path.join(download_folder, f"{filename}.zip")
        folder_path = os.path.join(download_folder, filename)

        if os.path.isfile(zip_path) or os.path.isdir(folder_path):
            if overwrite:
                if not logged_overwrite:
                    logger.info("Overwriting all existing feeds.")
                    logged_overwrite = True
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
            else:
                if not logged_skip:
                    logger.info("Skipping all already downloaded feeds.")
                    logged_skip = True
                existing = folder_path if os.path.isdir(folder_path) else zip_path
                zip_paths.append(os.path.abspath(existing))
                continue

        try:
            with requests.get(
                feed.download_url, headers=headers, stream=True, timeout=request_timeout
            ) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            zip_paths.append(os.path.abspath(zip_path))
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading feed '{filename}': {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue

    logger.info(f"Successfully downloaded {len(zip_paths)} feeds.")

    if not unzip:
        return zip_paths

    unzipped_count = 0
    result_paths = []
    for path in zip_paths:
        if path.endswith(".zip"):
            result_paths.append(io.unzip(path))
            os.remove(path)
            unzipped_count += 1
        else:
            result_paths.append(path)

    if unzipped_count > 0:
        logger.info(f"Successfully unzipped {unzipped_count} feeds.")

    return result_paths
