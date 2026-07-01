import pytest

from manhattan_reasoning_gym import _oauth


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_DEVICE = {
    "device_code": "dc",
    "user_code": "WXYZ-1234",
    "verification_uri": "https://github.com/login/device",
    "interval": 0,
}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(_oauth.time, "sleep", lambda *_: None)


def _seq(monkeypatch, *payloads):
    responses = iter([_Resp(p) for p in payloads])
    monkeypatch.setattr(_oauth.requests, "post", lambda *a, **k: next(responses))


def test_success_returns_token_and_prompts(monkeypatch):
    _seq(
        monkeypatch,
        _DEVICE,
        {"error": "authorization_pending"},
        {"access_token": "gho_x"},
    )
    prompts = []
    tok = _oauth.device_flow_token("cid", on_prompt=prompts.append)
    assert tok == "gho_x"
    assert any("WXYZ-1234" in p for p in prompts)  # user sees the code


def test_default_scope_is_least_privilege(monkeypatch):
    calls = []

    def fake_post(url, data=None, headers=None):
        calls.append((url, data))
        return _Resp(_DEVICE if "device/code" in url else {"access_token": "gho_x"})

    monkeypatch.setattr(_oauth.requests, "post", fake_post)
    _oauth.device_flow_token("cid", on_prompt=lambda m: None)

    device_data = next(d for (u, d) in calls if "device/code" in u)
    assert device_data["scope"] == ""  # allowlist needs no GitHub scope


def test_slow_down_then_success(monkeypatch):
    _seq(monkeypatch, _DEVICE, {"error": "slow_down"}, {"access_token": "gho_y"})
    assert _oauth.device_flow_token("cid", on_prompt=lambda m: None) == "gho_y"


def test_access_denied_raises(monkeypatch):
    _seq(monkeypatch, _DEVICE, {"error": "access_denied"})
    with pytest.raises(_oauth.DeviceFlowError, match="denied"):
        _oauth.device_flow_token("cid", on_prompt=lambda m: None)


def test_expired_token_raises(monkeypatch):
    _seq(monkeypatch, _DEVICE, {"error": "expired_token"})
    with pytest.raises(_oauth.DeviceFlowError, match="expired"):
        _oauth.device_flow_token("cid", on_prompt=lambda m: None)


def test_bad_device_code_response_raises(monkeypatch):
    _seq(monkeypatch, {"error": "invalid_client", "error_description": "bad client id"})
    with pytest.raises(_oauth.DeviceFlowError, match="bad client id"):
        _oauth.device_flow_token("cid", on_prompt=lambda m: None)
