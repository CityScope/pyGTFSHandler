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

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SHARED_FILE_ENV_VAR = "API_KEYS_FILE"
STANDARD_PATH = Path.home() / ".config" / "ptlos" / "api_keys.json"


def _load_keys(paths) -> Dict[str, Any]:
    """Every readable keys file merged; a file earlier in ``paths`` wins (unreadable files are skipped with a warning)."""
    merged: Dict[str, Any] = {}
    for path in reversed([Path(p) for p in paths]):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(data)
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable API keys file %s: %s", path, exc)
    return merged


def save_api_key(name: str, key: str, path: Optional[Path] = None) -> Path:
    """Store ``key`` under ``name`` in the standard keys file (``~/.config/ptlos/api_keys.json``, permissions 600), keeping the other keys.

    Args:
        name: Key name.
        key: The key.
        path: Keys file; default ``STANDARD_PATH``.

    Returns:
        The keys file.
    """
    path = Path(path) if path else STANDARD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = _load_keys([path])
    keys[name] = key
    path.write_text(json.dumps(keys, indent=1))
    path.chmod(0o600)
    return path

CONFIG_FILE_ENV_VAR = "PYGTFSHANDLER_API_KEYS_FILE"
DEFAULT_CONFIG_PATH = Path.home() / ".pygtfshandler" / "api_keys.json"
PACKAGE_KEYS_PATH = Path(__file__).resolve().parents[3] / "api_keys.json"   # older location: the pyGTFSHandler folder


def _candidate_config_paths() -> list:
    """Keys files, highest priority first: `API_KEYS_FILE`, `PYGTFSHANDLER_API_KEYS_FILE`, `~/.config/ptlos/api_keys.json`,
    then pyGTFSHandler's older locations (the package folder, the working directory and its parents,
    `~/.pygtfshandler/api_keys.json`)."""
    paths = [Path(v) for v in (os.getenv(SHARED_FILE_ENV_VAR), os.getenv(CONFIG_FILE_ENV_VAR)) if v]
    return paths + [STANDARD_PATH, PACKAGE_KEYS_PATH, *(d / "api_keys.json" for d in (Path.cwd(), *Path.cwd().parents)), DEFAULT_CONFIG_PATH]


def _load_api_keys_file() -> Dict[str, Any]:
    """Every readable keys file merged, earlier files winning.

    Returns:
        A dict mapping source name (e.g. "nap_es") to API key.
    """
    return _load_keys(_candidate_config_paths())


def get_api_key(source_name: str, api_key: Optional[str] = None, env_var: str = "") -> Optional[str]:
    """Resolve an API key for a downloader source: the explicit `api_key`, the `env_var` environment variable, then the keys files.

    Args:
        source_name: Key used to look up this source in the API keys file (a downloader's `SOURCE_NAME`).
        api_key: An explicitly provided API key, if any.
        env_var: Name of the environment variable that may hold this source's API key.

    Returns:
        The resolved API key, or `None` if it couldn't be found anywhere.
    """
    if api_key:
        return api_key
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)
    if not source_name:
        return None
    value = _load_api_keys_file().get(source_name)
    return str(value) if value else None
