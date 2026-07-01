import os
import stat

import pytest

from manhattan_reasoning_gym import _credentials


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the credentials store at a throwaway XDG config dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


URL = "https://api.example.com"


def test_load_missing_returns_none():
    assert _credentials.load(URL) is None


def test_save_then_load_roundtrip():
    _credentials.save(URL, "key-abc", "octocat")
    assert _credentials.load(URL) == "key-abc"


def test_save_uses_owner_only_permissions():
    path = _credentials.save(URL, "key-abc", "octocat")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_multiple_urls_are_independent():
    other = "https://api.other.com"
    _credentials.save(URL, "key-1", "alice")
    _credentials.save(other, "key-2", "bob")
    assert _credentials.load(URL) == "key-1"
    assert _credentials.load(other) == "key-2"


def test_save_overwrites_existing_url():
    _credentials.save(URL, "key-1", "alice")
    _credentials.save(URL, "key-2", "alice")
    assert _credentials.load(URL) == "key-2"


def test_clear_removes_only_target_url():
    other = "https://api.other.com"
    _credentials.save(URL, "key-1", "alice")
    _credentials.save(other, "key-2", "bob")
    assert _credentials.clear(URL) is True
    assert _credentials.load(URL) is None
    assert _credentials.load(other) == "key-2"


def test_clear_missing_returns_false():
    assert _credentials.clear(URL) is False


def test_corrupt_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "mrg" / "credentials.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ not json")
    assert _credentials.load(URL) is None
