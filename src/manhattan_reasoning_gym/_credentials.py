"""On-disk storage for Manhattan Reasoning Gym API keys.

Keys minted by ``mrg login`` are persisted to a small JSON file so the
user doesn't have to keep an API key in an environment variable. The file is
keyed by orchestrator URL, so a single machine can hold credentials for more
than one deployment.

Resolution order used elsewhere (``_cli._creds``): an explicit ``--api-key`` or
``$MRG_API_KEY`` always wins; this file is the fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(Path.home(), ".config")
    return Path(base) / "mrg"


def _path() -> Path:
    return _config_dir() / "credentials.json"


def _load_all() -> dict[str, dict[str, str]]:
    try:
        with open(_path(), encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load(api_url: str) -> str | None:
    """Return the stored API key for ``api_url``, or None if not logged in."""
    entry = _load_all().get(api_url)
    if not entry:
        return None
    return entry.get("api_key")


def save(api_url: str, api_key: str, github_username: str) -> Path:
    """Persist an API key for ``api_url`` with owner-only permissions."""
    data = _load_all()
    data[api_url] = {"api_key": api_key, "github_username": github_username}

    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then tighten perms; create with 0600 to avoid a readable window.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return path


def clear(api_url: str) -> bool:
    """Remove the stored API key for ``api_url``. Returns True if one existed."""
    data = _load_all()
    if api_url not in data:
        return False
    del data[api_url]

    path = _path()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return True
