"""Validated, comment-preserving writes back to the YAML config files.

Every edit goes through :func:`apply_edits` (or the watchlist helpers), which
write the change with ``ruamel.yaml`` (keeping the human comments intact),
then re-run :func:`load_config` so the *entire* config graph is re-validated
by the same pydantic models the rest of the platform trusts. If validation
fails (e.g. ``signal_weights`` no longer sums to 1.0) the original file is
restored and the error propagates — the on-disk config is never left invalid.

The dashboard is the only writer; config remains the single source of truth.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from trading_platform.core.config import AppConfig, load_config

# file_key -> filename. Mirrors load_config's four inputs.
FILES: dict[str, str] = {
    "settings": "settings.yaml",
    "weights": "weights.yaml",
    "risk": "risk_limits.yaml",
    "watchlist": "watchlist.yaml",
}


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # never line-wrap scalars
    y.indent(mapping=2, sequence=2, offset=0)
    return y


def _load(path: Path) -> Any:
    if not path.exists():
        return CommentedMap()
    with open(path, encoding="utf-8") as f:
        data = _yaml().load(f)
    return data if data is not None else CommentedMap()


def _atomic_dump(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            _yaml().dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _set_path(root: Any, dotted: str, value: Any) -> None:
    """Set a dotted path, replacing the leaf wholesale.

    Intermediate keys are treated as nested mappings (config sub-models such
    as ``llm`` or ``sizing``) and are navigated, never replaced — so their
    sibling keys and comments survive. The leaf value (scalar, list, or whole
    dict like ``signal_weights``) replaces whatever was there.
    """
    parts = dotted.split(".")
    node = root
    for p in parts[:-1]:
        child = node.get(p) if hasattr(node, "get") else None
        if not isinstance(child, dict):
            child = CommentedMap()
            node[p] = child
        node = child
    node[parts[-1]] = value


def _resolve_dir(config_dir: Path | str) -> Path:
    return Path(config_dir)


def apply_edits(
    config_dir: Path | str, file_key: str, updates: dict[str, Any]
) -> AppConfig:
    """Apply ``{dotted_path: value}`` edits to one config file, validated.

    Returns the freshly loaded, fully-validated AppConfig. On any validation
    error the file is rolled back to its previous contents and the exception
    is re-raised.
    """
    if file_key not in FILES:
        raise KeyError(f"unknown config file_key: {file_key!r}")
    config_dir = _resolve_dir(config_dir)
    path = config_dir / FILES[file_key]
    original = path.read_text(encoding="utf-8") if path.exists() else None

    data = _load(path)
    for dotted, value in updates.items():
        _set_path(data, dotted, value)
    _atomic_dump(path, data)

    try:
        return load_config(config_dir)
    except Exception:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original, encoding="utf-8")
        raise


def add_ticker(config_dir: Path | str, symbol: str, sector: str) -> AppConfig:
    """Append a ticker to watchlist.yaml (validated, rolled back on failure)."""
    symbol = symbol.strip().upper()
    sector = sector.strip()
    if not symbol:
        raise ValueError("symbol is required")
    if not sector:
        raise ValueError("sector is required")

    config_dir = _resolve_dir(config_dir)
    path = config_dir / FILES["watchlist"]
    original = path.read_text(encoding="utf-8") if path.exists() else None

    data = _load(path)
    tickers = data.get("tickers")
    if not isinstance(tickers, list):
        tickers = []
        data["tickers"] = tickers
    if any(isinstance(t, dict) and str(t.get("symbol")).upper() == symbol for t in tickers):
        raise ValueError(f"{symbol} is already on the watchlist")

    entry = CommentedMap()
    entry["symbol"] = symbol
    entry["sector"] = sector
    tickers.append(entry)
    _atomic_dump(path, data)

    try:
        return load_config(config_dir)
    except Exception:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original, encoding="utf-8")
        raise


def remove_ticker(config_dir: Path | str, symbol: str) -> AppConfig:
    """Remove a ticker from watchlist.yaml (validated, rolled back on failure)."""
    symbol = symbol.strip().upper()
    config_dir = _resolve_dir(config_dir)
    path = config_dir / FILES["watchlist"]
    original = path.read_text(encoding="utf-8") if path.exists() else None

    data = _load(path)
    tickers = data.get("tickers") or []
    kept = [t for t in tickers if not (isinstance(t, dict)
            and str(t.get("symbol")).upper() == symbol)]
    if len(kept) == len(tickers):
        raise ValueError(f"{symbol} is not on the watchlist")
    data["tickers"] = kept
    _atomic_dump(path, data)

    try:
        return load_config(config_dir)
    except Exception:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original, encoding="utf-8")
        raise
