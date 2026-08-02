"""Tests for `pyGTFSHandler.downloaders.utils.config`."""

import json

import pytest

from pyGTFSHandler.downloaders.utils import config


@pytest.fixture(autouse=True)
def _isolate_config_lookup(tmp_path, monkeypatch):
    """Point every candidate config path at empty/nonexistent locations.

    Prevents these tests from picking up a real `~/.pygtfshandler/api_keys.json`
    or a stray `api_keys.json` in the test runner's CWD.
    """
    monkeypatch.delenv(config.CONFIG_FILE_ENV_VAR, raising=False)
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", tmp_path / "home_config" / "api_keys.json")
    monkeypatch.chdir(tmp_path)
    yield


def test_explicit_api_key_wins_over_everything(monkeypatch):
    monkeypatch.setenv("SOME_ENV_VAR", "from-env")
    assert config.get_api_key("dummy", api_key="from-arg", env_var="SOME_ENV_VAR") == "from-arg"


def test_env_var_used_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv("SOME_ENV_VAR", "from-env")
    assert config.get_api_key("dummy", api_key=None, env_var="SOME_ENV_VAR") == "from-env"


def test_returns_none_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("SOME_ENV_VAR", raising=False)
    assert config.get_api_key("dummy", api_key=None, env_var="SOME_ENV_VAR") is None


def test_reads_key_from_cwd_config_file(tmp_path):
    (tmp_path / "api_keys.json").write_text(json.dumps({"dummy": "from-file"}))
    assert config.get_api_key("dummy", api_key=None, env_var="") == "from-file"


def test_reads_key_from_env_var_pointed_config_file(tmp_path, monkeypatch):
    custom_path = tmp_path / "custom" / "keys.json"
    custom_path.parent.mkdir()
    custom_path.write_text(json.dumps({"dummy": "from-custom-file"}))
    monkeypatch.setenv(config.CONFIG_FILE_ENV_VAR, str(custom_path))

    assert config.get_api_key("dummy", api_key=None, env_var="") == "from-custom-file"


def test_missing_source_in_file_returns_none(tmp_path):
    (tmp_path / "api_keys.json").write_text(json.dumps({"other_source": "x"}))
    assert config.get_api_key("dummy", api_key=None, env_var="") is None


def test_malformed_json_file_is_ignored_not_raised(tmp_path):
    (tmp_path / "api_keys.json").write_text("{not valid json")
    assert config.get_api_key("dummy", api_key=None, env_var="") is None


def test_env_var_takes_priority_over_file(tmp_path, monkeypatch):
    (tmp_path / "api_keys.json").write_text(json.dumps({"dummy": "from-file"}))
    monkeypatch.setenv("SOME_ENV_VAR", "from-env")
    assert config.get_api_key("dummy", api_key=None, env_var="SOME_ENV_VAR") == "from-env"


def test_home_and_cwd_files_are_merged_cwd_wins(tmp_path):
    home_config = tmp_path / "home_config" / "api_keys.json"
    home_config.parent.mkdir()
    home_config.write_text(json.dumps({"dummy": "from-home", "other": "home-only"}))
    (tmp_path / "api_keys.json").write_text(json.dumps({"dummy": "from-cwd"}))

    assert config.get_api_key("dummy", api_key=None, env_var="") == "from-cwd"
    assert config._load_api_keys_file()["other"] == "home-only"
