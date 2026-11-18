# -*- coding: utf-8 -*-
"""Resolving API keys without ever hardcoding them in the package.

Every downloader needs a secret (API key or refresh token) to talk to its
source. `BaseGTFSDownloader.__init__` resolves that secret through
`get_api_key`, trying each of the following in order and using the first
one found:

1. The `api_key` argument passed explicitly to the downloader's
   constructor -- the normal way for a user to supply a key at runtime
   without touching any file in this package.
2. The source-specific environment variable named by the downloader's
   `API_KEY_ENV_VAR` (e.g. `NAP_API_KEY`).
3. The keys file: one standard place for every key of every package of the project,
   `~/.config/ptlos/api_keys.json` (a JSON object `{"<source name>": "<key>"}`, outside every repository so it can never be
   committed), or the file named by `API_KEYS_FILE` (or `PYGTFSHANDLER_API_KEYS_FILE`). Older locations (`api_keys.json` in
   the pyGTFSHandler folder, the working directory or its parents, `~/.pygtfshandler/api_keys.json`) are still read.
   `api_keys.example.json` lists the key names.

None of these are required -- a downloader with no key available simply
gets `None` back, and raises its own `ValueError` if the source requires
one.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from geohierarchy import api_keys as _keys
from geohierarchy.api_keys import SHARED_FILE_ENV_VAR, STANDARD_PATH  # noqa: F401  (the lookup itself lives in geohierarchy)

logger = logging.getLogger(__name__)

CONFIG_FILE_ENV_VAR = "PYGTFSHANDLER_API_KEYS_FILE"
DEFAULT_CONFIG_PATH = Path.home() / ".pygtfshandler" / "api_keys.json"
PACKAGE_KEYS_PATH = Path(__file__).resolve().parents[3] / "api_keys.json"   # older location: the pyGTFSHandler folder


def _candidate_config_paths() -> list:
    """Keys files, highest priority first: geohierarchy's standard order (`API_KEYS_FILE`, `PYGTFSHANDLER_API_KEYS_FILE`,
    `~/.config/ptlos/api_keys.json`), then pyGTFSHandler's older locations (the package folder, the working directory and its
    parents, `~/.pygtfshandler/api_keys.json`)."""
    return _keys.candidate_paths(
        env_vars=[CONFIG_FILE_ENV_VAR],
        files=[PACKAGE_KEYS_PATH, *(d / "api_keys.json" for d in (Path.cwd(), *Path.cwd().parents)), DEFAULT_CONFIG_PATH],
        standard_path=STANDARD_PATH,
    )


def _load_api_keys_file() -> Dict[str, Any]:
    """Every readable keys file merged, earlier files winning (see `geohierarchy.api_keys.load_keys`).

    Returns:
        A dict mapping source name (e.g. "nap_es") to API key.
    """
    return _keys.load_keys(_candidate_config_paths())


def get_api_key(source_name: str, api_key: Optional[str] = None, env_var: str = "") -> Optional[str]:
    """Resolve an API key for a downloader source: the explicit `api_key`, the `env_var` environment variable, then the keys files.

    Args:
        source_name: Key used to look up this source in the API keys file (a downloader's `SOURCE_NAME`).
        api_key: An explicitly provided API key, if any.
        env_var: Name of the environment variable that may hold this source's API key.

    Returns:
        The resolved API key, or `None` if it couldn't be found anywhere.
    """
    if not source_name:
        return api_key or (_keys.os.getenv(env_var) if env_var else None) or None
    return _keys.resolve_api_key(source_name, env_var or None, api_key, paths=_candidate_config_paths())
