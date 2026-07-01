import pytest

import manhattan_reasoning_gym
from manhattan_reasoning_gym import _client, _credentials


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MRG_API_KEY", raising=False)
    return tmp_path


def _app(**kw):
    return manhattan_reasoning_gym.App("x", design="d.py", fpga_id=0, **kw)


def test_explicit_arg_wins():
    monkey_url = _client.DEFAULT_API_URL
    _credentials.save(monkey_url, "from-store", "octocat")
    assert _app(api_key="from-arg").api_key == "from-arg"


def test_env_beats_store(monkeypatch):
    monkeypatch.setenv("MRG_API_KEY", "from-env")
    _credentials.save(_client.DEFAULT_API_URL, "from-store", "octocat")
    assert _app().api_key == "from-env"


def test_falls_back_to_stored_login():
    _credentials.save(_client.DEFAULT_API_URL, "from-store", "octocat")
    assert _app().api_key == "from-store"


def test_store_lookup_is_per_api_url():
    other = "https://api.other.com"
    _credentials.save(other, "other-key", "octocat")
    # Default-URL app finds nothing; the key belongs to a different URL.
    assert _app().api_key == ""
    assert _app(api_url=other).api_key == "other-key"


def test_empty_when_nothing_configured():
    assert _app().api_key == ""
