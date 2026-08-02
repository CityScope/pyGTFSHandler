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
3. A local, gitignored JSON file mapping source names to keys, so a key
   doesn't have to be re-typed or re-exported every session. The file's
   location is `PYGTFSHANDLER_API_KEYS_FILE` if that environment variable
   is set, otherwise `~/.pygtfshandler/api_keys.json`, otherwise
   `api_keys.json` in the current working directory. `api_keys.example.json`
   at the repository root documents the expected shape; the real
   `api_keys.json` is listed in `.gitignore` so keys never get committed.

None of these are required -- a downloader with no key available simply
gets `None` back, and raises its own `ValueError` if the source requires
one.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_FILE_ENV_VAR = "PYGTFSHANDLER_API_KEYS_FILE"
DEFAULT_CONFIG_PATH = Path.home() / ".pygtfshandler" / "api_keys.json"


def _candidate_config_paths() -> list:
    """Build the ordered list of paths to look for an API keys file in.

    Returns:
        Candidate paths, in priority order (highest first): the path from
        `PYGTFSHANDLER_API_KEYS_FILE` (if set), `api_keys.json` in the
        current working directory (e.g. a project-local file), and the
        user's home config file.
    """
    paths = []
    env_path = os.getenv(CONFIG_FILE_ENV_VAR)
    if env_path:
        paths.append(Path(env_path))
    paths.append(Path.cwd() / "api_keys.json")
    paths.append(DEFAULT_CONFIG_PATH)
    return paths


def _load_api_keys_file() -> Dict[str, Any]:
    """Load and merge every existing API keys file, in priority order.

    Later (lower-priority) files fill in keys missing from earlier ones,
    so a home-directory file and a per-project file can coexist.

    Returns:
        A dict mapping source name (e.g. "nap_es") to API key. Empty if
        no config file exists or none could be parsed.
    """
    merged: Dict[str, Any] = {}
    for path in reversed(_candidate_config_paths()):
        if not path.is_file():
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged.update(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read API keys file '{path}': {e}")
    return merged


def get_api_key(
    source_name: str,
    api_key: Optional[str] = None,
    env_var: str = "",
) -> Optional[str]:
    """Resolve an API key for a downloader source.

    Tries, in order: the explicitly passed `api_key`, the `env_var`
    environment variable, and the `source_name` entry of the local API
    keys file (see module docstring for its lookup locations).

    Args:
        source_name: Key used to look up this source in the API keys
            file (a downloader's `SOURCE_NAME`).
        api_key: An explicitly provided API key, if any.
        env_var: Name of the environment variable that may hold this
            source's API key.

    Returns:
        The resolved API key, or `None` if it couldn't be found anywhere.
    """
    if api_key:
        return api_key

    if env_var:
        from_env = os.getenv(env_var)
        if from_env:
            return from_env

    if source_name:
        from_file = _load_api_keys_file().get(source_name)
        if from_file:
            return str(from_file)

    return None
