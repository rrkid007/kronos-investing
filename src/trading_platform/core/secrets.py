"""Secret (API key) storage via a gitignored ``.env`` file.

Secrets are NEVER written to the YAML config (which is git-tracked). The config
only stores the *names* of the environment variables an agent reads
(``FINNHUB_API_KEY``, ``MEMO_LLM_API_KEY``, ``ALPACA_API_KEY_ID``, ...); this
module manages their *values* in a single ``.env`` file (chmod 600), loads them
into the process environment at startup, and lets the dashboard update them
live — so a key entered in the UI takes effect without a restart.

Values are never rendered back to the UI: the dashboard only ever shows whether
a given variable is currently set.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env_file(path: Path | str, *, override: bool = True) -> int:
    """Load ``KEY=value`` lines from ``path`` into ``os.environ``.

    Returns the number of variables applied. A missing or unreadable file is a
    no-op (0) and never raises — secrets loading must never take down the
    dashboard; the worst case is that keys simply aren't set.
    With ``override=False`` an already-set variable is left untouched.
    """
    path = Path(path)
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read env file %s: %s", path, exc)
        return 0
    n = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
            n += 1
    return n


def _parse(path: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Return (line-order tokens, key->value). Tokens preserve comments/blanks."""
    order: list[tuple[str, str]] = []
    entries: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                order.append(("raw", raw))
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            entries[key] = val.strip()
            order.append(("kv", key))
    return order, entries


def write_secret(path: Path | str, key: str, value: str) -> None:
    """Upsert ``KEY=value`` in the ``.env`` file and live ``os.environ``.

    An empty value removes the key. The file is written atomically and chmod'd
    to 0600. Existing comments and unrelated keys are preserved.
    """
    key = key.strip()
    if not key:
        raise ValueError("env var name is required")
    path = Path(path)
    order, entries = _parse(path)

    remove = value is None or value.strip() == ""
    if remove:
        entries.pop(key, None)
    else:
        entries[key] = value.strip()

    lines: list[str] = []
    seen: set[str] = set()
    for kind, payload in order:
        if kind == "raw":
            lines.append(payload)
        elif payload in entries and payload not in seen:
            lines.append(f"{payload}={entries[payload]}")
            seen.add(payload)
        # a removed key's token is simply skipped
    for k, v in entries.items():
        if k not in seen:
            lines.append(f"{k}={v}")
            seen.add(k)

    text = ("\n".join(lines).rstrip("\n") + "\n") if lines else ""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    if remove:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value.strip()


def is_set(env_name: str) -> bool:
    """True if the named environment variable currently holds a non-empty value."""
    return bool(os.environ.get(env_name, "").strip())
