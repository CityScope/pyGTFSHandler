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


BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _legacy_tls_session() -> requests.Session:
    """A session accepting weak/old TLS configurations (no certificate verification, cipher security level 0)."""
    import ssl

    from requests.adapters import HTTPAdapter

    class _Adapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            try:
                ctx.set_ciphers("DEFAULT@SECLEVEL=0")
            except ssl.SSLError:
                pass
            kwargs["ssl_context"] = ctx
            return super().init_poolmanager(*args, **kwargs)

    session = requests.Session()
    session.mount("https://", _Adapter())
    return session


def robust_get(url: str, timeout: int = 60, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    """Streamed GET that survives the usual agency-site quirks.

    Sends a browser User-Agent (many transit sites answer 403/406 to the default
    python-requests one), retries connection resets, and, as a last resort for
    broken certificate chains, retries once without certificate verification.
    Raises `requests.RequestException` only after all of that failed. The caller
    must close/`with` the returned response.
    """
    import time

    merged = {"User-Agent": BROWSER_USER_AGENT, **(headers or {})}
    last: Optional[Exception] = None
    verify = True
    session = requests
    for attempt in range(5):
        try:
            response = session.get(url, headers=merged, stream=True, timeout=timeout, allow_redirects=True, verify=verify)
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError as exc:
            last = exc
            if verify:
                logger.warning(f"SSL verification failed for {url}; retrying without verification.")
                verify = False
            elif session is requests:
                # Old servers with weak TLS settings (e.g. NCTD): relax the cipher security level too.
                logger.warning(f"SSL handshake failed for {url}; retrying with legacy TLS settings.")
                session = _legacy_tls_session()
        except requests.exceptions.HTTPError as exc:
            last = exc
            if exc.response is not None and exc.response.status_code < 500:
                break  # 401/403/404...: retrying won't help
            time.sleep(2 * (attempt + 1))
        except requests.exceptions.RequestException as exc:  # resets, timeouts
            last = exc
            time.sleep(2 * (attempt + 1))
    assert last is not None
    raise last


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
            with robust_get(feed.download_url, timeout=request_timeout, headers=headers) as r:
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
            try:
                result_paths.append(io.unzip(path))
            except Exception as e:
                logger.error(f"Skipping unreadable feed '{path}': {e}")
                if os.path.exists(path):
                    os.remove(path)
                continue
            os.remove(path)
            unzipped_count += 1
        else:
            result_paths.append(path)

    if unzipped_count > 0:
        logger.info(f"Successfully unzipped {unzipped_count} feeds.")

    return result_paths
