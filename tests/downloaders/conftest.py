"""Test isolation for the `downloaders` test suite.

These tests must never pick up a real API key -- from the environment or
from a local `api_keys.json` -- since that could turn a "no key configured"
test into an accidental live network call. This autouse fixture clears the
known API key environment variables and points the config file lookup
(`downloaders.utils.config`) at an empty, per-test directory before every
test in this package.
"""

from pathlib import Path

import pytest

from pyGTFSHandler.downloaders.utils import config

_API_KEY_ENV_VARS = [
    "NAP_API_KEY",
    "TRANSITLAND_API_KEY",
    "MOBILITY_DATABASE_REFRESH_TOKEN",
    "SOCRATA_APP_TOKEN",
    config.CONFIG_FILE_ENV_VAR,
    config.SHARED_FILE_ENV_VAR,
]


@pytest.fixture(autouse=True)
def _isolate_downloader_secrets(tmp_path, monkeypatch):
    for var in _API_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", tmp_path / "unused_home" / "api_keys.json")
    # the standard ~/.config/ptlos/api_keys.json and the package-folder copy point at empty per-test paths too
    monkeypatch.setattr(config, "STANDARD_PATH", tmp_path / "unused_standard" / "api_keys.json")
    monkeypatch.setattr(config, "PACKAGE_KEYS_PATH", tmp_path / "unused_package" / "api_keys.json")
    monkeypatch.chdir(tmp_path)
    yield
