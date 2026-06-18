"""Introspect the pydantic config models into editable form metadata.

The dashboard renders *every* configurable field without hand-maintaining a
form: walk the models, classify each field, and emit a grouped schema the
template iterates over. The same metadata drives type coercion when a form is
submitted, so what the UI shows and what the writer accepts never drift.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from trading_platform.core.config import AppConfig

# kinds: bool | int | float | str | choice | list | map | section


@dataclass
class Field:
    path: str          # dotted path within the file (e.g. "llm.model")
    label: str
    kind: str
    value: str         # render-ready string (textarea/input value)
    choices: list[str] | None = None
    map_value_kind: str = "str"  # for kind == "map": coercion of values


@dataclass
class Section:
    id: str
    title: str
    fields: list[Field] = dc_field(default_factory=list)
    note: str = ""


@dataclass
class FileGroup:
    file_key: str
    title: str
    sections: list[Section] = dc_field(default_factory=list)


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in text.strip().lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "general"


def _classify(annotation: Any) -> tuple[str, Any]:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        return "choice", [str(a) for a in args]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _classify(non_none[0])
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return "section", annotation
    if origin in (list, typing.List):  # noqa: UP006
        return "list", None
    if origin in (dict, typing.Dict):  # noqa: UP006
        vt = args[1] if len(args) > 1 else str
        vkind, _ = _classify(vt)
        return "map", vkind
    if annotation is bool:
        return "bool", None
    if annotation is int:
        return "int", None
    if annotation is float:
        return "float", None
    if annotation in (str, Path):
        return "str", None
    return "str", None


def _display(kind: str, value: Any) -> str:
    if value is None:
        return ""
    if kind == "bool":
        return "true" if value else "false"
    if kind == "list":
        return "\n".join(str(v) for v in value)
    if kind == "map":
        return "\n".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def _make_field(name: str, annotation: Any, value: Any, prefix: str) -> Field:
    kind, extra = _classify(annotation)
    return Field(
        path=f"{prefix}{name}",
        label=_humanize(name),
        kind=kind,
        value=_display(kind, value),
        choices=extra if kind == "choice" else None,
        map_value_kind=extra if kind == "map" else "str",
    )


def _walk(model: BaseModel, prefix: str) -> tuple[list[Field], list[tuple[str, BaseModel, str]]]:
    """Return (scalar/leaf fields here, [(name, submodel_instance, dotted_prefix)])."""
    scalars: list[Field] = []
    submodels: list[tuple[str, BaseModel, str]] = []
    for name, info in type(model).model_fields.items():
        annotation = info.annotation
        value = getattr(model, name)
        kind, _ = _classify(annotation)
        if kind == "section" and isinstance(value, BaseModel):
            submodels.append((name, value, f"{prefix}{name}."))
        else:
            scalars.append(_make_field(name, annotation, value, prefix))
    return scalars, submodels


def _sections_for(model: BaseModel, general_title: str, prefix: str = "") -> list[Section]:
    scalars, submodels = _walk(model, prefix)
    sections: list[Section] = []
    if scalars:
        sections.append(Section(id=_slug(prefix or general_title),
                                title=general_title, fields=scalars))
    for name, sub, sub_prefix in submodels:
        sub_scalars, nested = _walk(sub, sub_prefix)
        sections.append(Section(id=_slug(sub_prefix), title=_humanize(name),
                                fields=sub_scalars))
        # one level of nesting (e.g. risk.sizing) — flatten as its own section
        for nname, nsub, nprefix in nested:
            nscalars, _ = _walk(nsub, nprefix)
            sections.append(Section(id=_slug(nprefix),
                                    title=f"{_humanize(name)} · {_humanize(nname)}",
                                    fields=nscalars))
    return sections


def build_schema(config: AppConfig) -> list[FileGroup]:
    """Full editable schema for settings.yaml, weights.yaml, risk_limits.yaml."""
    groups: list[FileGroup] = []

    settings_sections = _sections_for(config.settings, "General")
    groups.append(FileGroup("settings", "Platform Settings", settings_sections))

    weights_sections = _sections_for(config.weights, "Signal Weights")
    for s in weights_sections:
        if any(f.path == "signal_weights" for f in s.fields):
            s.note = "Signal weights must sum to 1.0 (validated on save)."
    groups.append(FileGroup("weights", "Weights & Thresholds", weights_sections))

    risk_sections = _sections_for(config.risk, "Limits")
    groups.append(FileGroup("risk", "Risk Limits", risk_sections))

    return groups


def coercion_map(config: AppConfig) -> dict[str, dict[str, Field]]:
    """{file_key: {dotted_path: Field}} used to coerce submitted form values."""
    out: dict[str, dict[str, Field]] = {}
    for group in build_schema(config):
        index: dict[str, Field] = {}
        for section in group.sections:
            for f in section.fields:
                index[f.path] = f
        out[group.file_key] = index
    return out


def coerce(field: Field, raw: str) -> Any:
    """Coerce a submitted string into the typed value the YAML writer expects."""
    raw = raw if raw is not None else ""
    if field.kind == "bool":
        return str(raw).strip().lower() in ("true", "1", "yes", "on")
    if field.kind == "int":
        return int(float(raw)) if str(raw).strip() != "" else 0
    if field.kind == "float":
        return float(raw) if str(raw).strip() != "" else 0.0
    if field.kind == "choice":
        return str(raw)
    if field.kind == "list":
        items = [line.strip() for line in str(raw).replace(",", "\n").splitlines()]
        return [i for i in items if i]
    if field.kind == "map":
        result: dict[str, Any] = {}
        for line in str(raw).splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if not k:
                continue
            if field.map_value_kind == "float":
                result[k] = float(v)
            elif field.map_value_kind == "int":
                result[k] = int(float(v))
            else:
                result[k] = v
        return result
    return str(raw)
