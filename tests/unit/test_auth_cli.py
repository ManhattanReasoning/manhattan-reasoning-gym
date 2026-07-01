import argparse

import pytest
import requests

from manhattan_reasoning_gym import _cli, _client, _credentials

URL = "https://api.test"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    from manhattan_reasoning_gym import _oauth

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MRG_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("MRG_GITHUB_CLIENT_ID", raising=False)
    # Neutralize the baked-in client id so tests pick the path explicitly and
    # never touch real GitHub; device-flow tests set client_id themselves.
    monkeypatch.setattr(_oauth, "DEFAULT_CLIENT_ID", "")
    return tmp_path


def _ns(**kw):
    kw.setdefault("api_url", URL)
    kw.setdefault("api_key", None)
    kw.setdefault("client_id", None)
    kw.setdefault("github_token", None)
    return argparse.Namespace(**kw)


# ── login ──────────────────────────────────────────────────────────────────

def test_login_with_flag_stores_key(monkeypatch, capsys):
    seen = {}

    def fake_exchange(token, url):
        seen["token"], seen["url"] = token, url
        return {"api_key": "mint-abc", "github_username": "octocat"}

    monkeypatch.setattr(_client, "exchange_github_token", fake_exchange)

    _cli.cmd_login(_ns(github_token="ghp_fake"))

    assert seen == {"token": "ghp_fake", "url": URL}
    assert _credentials.load(URL) == "mint-abc"
    assert "octocat" in capsys.readouterr().out


def test_login_reads_github_token_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_env")
    captured = {}
    monkeypatch.setattr(
        _client, "exchange_github_token",
        lambda token, url: captured.update(token=token)
        or {"api_key": "k", "github_username": "u"},
    )
    _cli.cmd_login(_ns(github_token=None))
    assert captured["token"] == "ghp_env"


def test_login_no_token_exits(monkeypatch):
    # No flag, no env, no client id, and an empty prompt.
    monkeypatch.setattr(_cli.getpass, "getpass", lambda *_: "")
    with pytest.raises(SystemExit):
        _cli.cmd_login(_ns(github_token=None))


def test_login_uses_device_flow_when_client_id_set(monkeypatch):
    from manhattan_reasoning_gym import _oauth

    seen = {}
    monkeypatch.setattr(_oauth, "device_flow_token", lambda cid, **k: "gho_oauth")

    def fake_exchange(token, url):
        seen["token"] = token
        return {"api_key": "mint", "github_username": "octocat"}

    monkeypatch.setattr(_client, "exchange_github_token", fake_exchange)

    _cli.cmd_login(_ns(client_id="cid123"))

    assert seen["token"] == "gho_oauth"           # OAuth token was exchanged
    assert _credentials.load(URL) == "mint"


def test_login_pat_flag_takes_precedence_over_device_flow(monkeypatch):
    from manhattan_reasoning_gym import _oauth

    def boom(*a, **k):
        raise AssertionError("device flow should not run when a PAT is given")

    monkeypatch.setattr(_oauth, "device_flow_token", boom)
    captured = {}
    monkeypatch.setattr(
        _client, "exchange_github_token",
        lambda t, u: captured.update(token=t)
        or {"api_key": "k", "github_username": "u"},
    )

    _cli.cmd_login(_ns(github_token="ghp_pat", client_id="cid123"))
    assert captured["token"] == "ghp_pat"


def test_login_exchange_failure_exits_and_stores_nothing(monkeypatch):
    resp = requests.Response()
    resp.status_code = 403

    def boom(token, url):
        raise requests.HTTPError(response=resp)

    monkeypatch.setattr(_client, "exchange_github_token", boom)

    with pytest.raises(SystemExit):
        _cli.cmd_login(_ns(github_token="ghp_fake"))
    assert _credentials.load(URL) is None


# ── logout ─────────────────────────────────────────────────────────────────

def test_logout_revokes_and_clears(monkeypatch):
    _credentials.save(URL, "mint-abc", "octocat")
    revoked = {}
    monkeypatch.setattr(
        _client, "revoke_key",
        lambda key, url: revoked.update(key=key, url=url),
    )

    _cli.cmd_logout(_ns())

    assert revoked == {"key": "mint-abc", "url": URL}
    assert _credentials.load(URL) is None


def test_logout_when_not_logged_in_is_noop(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(
        _client, "revoke_key",
        lambda key, url: called.__setitem__("n", called["n"] + 1),
    )
    _cli.cmd_logout(_ns())
    assert called["n"] == 0


def test_logout_clears_even_if_revoke_fails(monkeypatch, capsys):
    _credentials.save(URL, "mint-abc", "octocat")

    def boom(key, url):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(_client, "revoke_key", boom)

    _cli.cmd_logout(_ns())

    assert _credentials.load(URL) is None
    assert "warning" in capsys.readouterr().out.lower()


# ── credential resolution ────────────────────────────────────────────────────

def test_creds_prefers_flag_over_env_and_store(monkeypatch):
    monkeypatch.setenv("MRG_API_KEY", "from-env")
    _credentials.save(URL, "from-store", "octocat")
    key, url = _cli._creds(_ns(api_key="from-flag"))
    assert key == "from-flag"
    assert url == URL


def test_creds_prefers_env_over_store(monkeypatch):
    monkeypatch.setenv("MRG_API_KEY", "from-env")
    _credentials.save(URL, "from-store", "octocat")
    key, _ = _cli._creds(_ns(api_key=None))
    assert key == "from-env"


def test_creds_falls_back_to_store():
    _credentials.save(URL, "from-store", "octocat")
    key, _ = _cli._creds(_ns(api_key=None))
    assert key == "from-store"


def test_creds_exits_when_no_key():
    with pytest.raises(SystemExit):
        _cli._creds(_ns(api_key=None))
