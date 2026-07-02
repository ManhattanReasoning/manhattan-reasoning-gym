"""GitHub OAuth device flow for ``mrg login``.

Lets a user authenticate in their browser instead of pasting a personal access
token: we ask GitHub for a device code, show the user a URL + short code, and
poll until they approve. The resulting GitHub access token is then exchanged for
our opaque API key (see ``_client.exchange_github_token``) and discarded.

The ``client_id`` identifies a GitHub OAuth App with **Device Flow enabled**.
For device flow it is public (not a secret), so it can be hardcoded here or
supplied via ``--client-id`` / ``$MRG_GITHUB_CLIENT_ID``. Until an app is
registered, ``DEFAULT_CLIENT_ID`` is empty and ``mrg login`` falls back to PAT.
"""

from __future__ import annotations

import time

import requests

_DEVICE_CODE_URL = "https://github.com/login/device/code"
_TOKEN_URL = "https://github.com/login/oauth/access_token"

# Public client_id of the registered GitHub OAuth App (device flow enabled). This
# is public, not a secret. Override per-call via --client-id / $MRG_GITHUB_CLIENT_ID.
DEFAULT_CLIENT_ID = "Ov23liqcHWiySNlvlK6P"

# Requested scope. Access is gated by GITHUB_USER_ALLOWLIST (by username), which
# only needs the user's login — readable with no scope. Keep this empty for least
# privilege so the consent screen asks for nothing extra. Bump to "read:org" only
# if you later switch to org-membership gating.
DEFAULT_SCOPE = ""


class DeviceFlowError(RuntimeError):
    """Raised when the GitHub device flow can't complete."""


def device_flow_token(
    client_id: str,
    scope: str = DEFAULT_SCOPE,
    *,
    on_prompt=None,
) -> str:
    """Run GitHub's device flow and return a GitHub access token.

    Prints (or passes to ``on_prompt``) the verification URL and user code, then
    polls until the user authorizes. Raises ``DeviceFlowError`` on denial,
    expiry, or a GitHub-side error.
    """
    resp = requests.post(
        _DEVICE_CODE_URL,
        data={"client_id": client_id, "scope": scope},
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    if "device_code" not in data:
        raise DeviceFlowError(data.get("error_description") or str(data))

    device_code = data["device_code"]
    interval = int(data.get("interval", 5))
    prompt = (
        f"Open {data['verification_uri']} and enter code: {data['user_code']}"
    )
    if on_prompt:
        on_prompt(prompt)
    else:
        # flush: when stdout isn't a tty the prompt must not sit in the buffer
        # while we block polling GitHub below.
        print(prompt, flush=True)

    while True:
        time.sleep(interval)
        resp = requests.post(
            _TOKEN_URL,
            data={
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        body = resp.json()
        error = body.get("error")
        if not error:
            return body["access_token"]
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "expired_token":
            raise DeviceFlowError("device code expired — run `mrg login` again")
        if error == "access_denied":
            raise DeviceFlowError("authorization was denied")
        raise DeviceFlowError(body.get("error_description") or error)
