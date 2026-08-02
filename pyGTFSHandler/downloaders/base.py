# -*- coding: utf-8 -*-
"""Common interface that every GTFS source downloader implements.

This module defines `BaseGTFSDownloader`, the abstract base class that all
downloaders in this package (`downloaders.mobility_database`,
`downloaders.transitland`, `downloaders.spain.nap`, ...) inherit from. The
goal is uniformity: regardless of which catalog a downloader talks to, it
should be constructed the same way (an optional API key, falling back to an
environment variable specific to that source), expose the same
`search_feeds()` / `download_feeds()` shape, and produce results
(`downloaders.utils.models.GTFSFeedMetadata`) that can be handed to any
other downloader's `download_feeds()` unchanged.

Concrete subclasses only need to implement `search_feeds()` -- translating
their source's filters and response shape into a list of
`GTFSFeedMetadata` -- since `download_feeds()` is provided here via
`downloaders.utils.download.download_feeds`.

The API key itself is never read from a file hardcoded in this package:
`__init__` resolves it via `downloaders.utils.config.get_api_key`, which
checks (in order) the explicit `api_key` argument, the `API_KEY_ENV_VAR`
environment variable, and an optional local, gitignored API keys file --
see that module's docstring for the full lookup order and file format.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .utils.config import get_api_key
from .utils.download import download_feeds
from .utils.models import GTFSFeedMetadata


class BaseGTFSDownloader(ABC):
    """Abstract base class for a GTFS catalog/source downloader client.

    Subclasses must implement `search_feeds()`. `download_feeds()` is
    shared across all sources and downloads whatever `GTFSFeedMetadata`
    list is passed to it.

    Attributes:
        api_key: The API key used to authenticate with the source, or
            None if the source requires no authentication.
    """

    #: Name of the environment variable holding the API key for this
    #: source. Subclasses should override this (e.g. "NAP_API_KEY").
    API_KEY_ENV_VAR: str = ""

    #: Short identifier for this source, stored on every
    #: `GTFSFeedMetadata.source` this downloader produces.
    SOURCE_NAME: str = ""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the downloader with an API key.

        Args:
            api_key: API key for the source. If not provided, it is
                resolved via `downloaders.utils.config.get_api_key`: the
                `API_KEY_ENV_VAR` environment variable, then a local API
                keys file, if either is set up.
        """
        self.api_key = get_api_key(self.SOURCE_NAME, api_key, self.API_KEY_ENV_VAR)

    @abstractmethod
    def search_feeds(self, *args, **kwargs) -> List[GTFSFeedMetadata]:
        """Search the source's catalog for GTFS feeds.

        Returns:
            A list of `GTFSFeedMetadata` describing the matching feeds.
        """
        raise NotImplementedError

    def download_feeds(
        self,
        feeds: List[GTFSFeedMetadata],
        download_folder: str,
        overwrite: bool = False,
        unzip: bool = True,
    ) -> List[str]:
        """Download previously found feeds to a local folder.

        Args:
            feeds: Feeds to download, as returned by `search_feeds()`.
            download_folder: Directory to store (and, if `unzip`,
                extract) the downloaded feeds into.
            overwrite: If True, re-download and replace files that
                already exist on disk.
            unzip: If True, extract each ZIP after downloading and delete
                the ZIP file.

        Returns:
            Absolute paths to the downloaded feeds (ZIP files, or
            extracted folders if `unzip` is True).
        """
        return download_feeds(
            feeds, download_folder, overwrite=overwrite, unzip=unzip
        )
