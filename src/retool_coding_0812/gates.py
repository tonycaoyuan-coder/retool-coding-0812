"""Reproducible protocol fingerprints and hard pre-training gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .data import file_sha256, stable_hash
from .protocol import PROMPT_STRATEGIES, TOOLS, system_prompt


PROTOCOL_FIELDS = (
    "max_assistant_tokens",
    "max_trajectory_tokens",
    "max_tool_calls",
    "max_assistant_turns",
    "max_tool_response_tokens",
)


def protocol_payload(values: Mapping[str, Any]) -> dict[str, int]:
    payload = {field: int(values[field]) for field in PROTOCOL_FIELDS}
    if min(payload.values()) < 1:
        raise ValueError(f"Protocol fields must be positive: {payload}")
    if payload["max_assistant_turns"] < payload["max_tool_calls"] + 1:
        raise ValueError("max_assistant_turns must leave a final turn after tool calls")
    return payload


def prompt_fingerprints(protocol: Mapping[str, Any]) -> dict[str, str]:
    max_tool_calls = int(protocol["max_tool_calls"])
    return {
        variant: stable_hash(system_prompt(variant, max_tool_calls=max_tool_calls))
        for variant in sorted(PROMPT_STRATEGIES)
    }


def experiment_fingerprint(
    *,
    base_model: str,
    dataset_manifest_sha256: str,
    protocol: Mapping[str, Any],
    model_fingerprint: str = "base",
) -> tuple[str, dict[str, Any]]:
    normalized = protocol_payload(protocol)
    payload = {
        "base_model": base_model,
        "model_fingerprint": model_fingerprint,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "protocol": normalized,
        "prompt_sha256": prompt_fingerprints(normalized),
        "tools_sha256": stable_hash(TOOLS),
    }
    return stable_hash(payload), payload


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Required manifest is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must be a JSON object: {source}")
    return value


def validate_preflight(path: str | Path, *, image: str, image_id: str) -> dict[str, Any]:
    manifest = load_json(path)
    checks = manifest.get("checks") or {}
    if manifest.get("image") != image or manifest.get("image_id") != image_id:
        raise ValueError("Preflight manifest image fingerprint does not match Docker runtime")
    if not checks or not all(bool(value) for value in checks.values()):
        raise ValueError("Preflight manifest contains failed or missing checks")
    return manifest


def validate_data_manifest(data_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    data = Path(data_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = load_json(manifest_file)
    expected_sha = file_sha256(data)
    matches = []
    for split, raw in (manifest.get("splits") or {}).items():
        item = dict(raw)
        candidate = (manifest_file.parent / str(item.get("file", ""))).resolve()
        if candidate == data:
            matches.append((str(split), item))
    if len(matches) != 1:
        raise ValueError(f"Data file is not uniquely declared by manifest: {data}")
    split, item = matches[0]
    if item.get("file_sha256") != expected_sha:
        raise ValueError(f"Data SHA-256 differs from manifest for split {split}")
    if int(item.get("count", -1)) < 1:
        raise ValueError(f"Data manifest has invalid count for split {split}")
    return {
        "path": str(data),
        "split": split,
        "file_sha256": expected_sha,
        "manifest_path": str(manifest_file),
        "manifest_sha256": file_sha256(manifest_file),
        "count": int(item["count"]),
    }


def validate_training_gate(
    smoke_path: str | Path,
    *,
    base_model: str,
    dataset_manifest_sha256: str,
    protocol: Mapping[str, Any],
    prompt_variant: str,
    model_fingerprint: str = "base",
) -> dict[str, Any]:
    smoke = load_json(smoke_path)
    if not smoke.get("passed"):
        raise ValueError("Smoke gate failed; refusing to create TrainingClient")
    expected, payload = experiment_fingerprint(
        base_model=base_model,
        dataset_manifest_sha256=dataset_manifest_sha256,
        protocol=protocol,
        model_fingerprint=model_fingerprint,
    )
    if smoke.get("experiment_fingerprint") != expected:
        raise ValueError("Smoke manifest model/data/prompt/tool/protocol fingerprint mismatch")
    if prompt_variant not in (smoke.get("variants") or {}):
        raise ValueError(f"Smoke manifest does not cover prompt variant {prompt_variant}")
    if smoke.get("fingerprint_payload") != payload:
        raise ValueError("Smoke manifest fingerprint payload is inconsistent")
    return smoke


def validate_selected_protocol(
    config_path: str | Path,
    *,
    smoke_path: str | Path,
    base_model: str,
    dataset_manifest_sha256: str,
    protocol: Mapping[str, Any],
    model_fingerprint: str,
) -> dict[str, Any]:
    source = Path(config_path)
    if not source.exists():
        raise ValueError(f"Selected protocol config is missing: {source}")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Protocol config root must be a mapping")
    expected_protocol = protocol_payload(protocol)
    smoke = load_json(smoke_path)
    try:
        control = dict(value["models"][0]["metadata"]["selected_control"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("MyEval selected config lacks model selected_control metadata") from exc
    if control.get("status") != "selected":
        raise ValueError("Protocol config is not marked selected")
    expected = {
        "base_model": base_model,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "protocol": expected_protocol,
        "model_fingerprint": model_fingerprint,
        "smoke_manifest_sha256": file_sha256(smoke_path),
        "smoke_experiment_fingerprint": smoke.get("experiment_fingerprint"),
    }
    actual = {
        "base_model": value["models"][0].get("base_model"),
        "dataset_manifest_sha256": control.get("dataset_manifest_sha256"),
        "protocol": control.get("protocol"),
        "model_fingerprint": control.get("model_recipe_fingerprint"),
        "smoke_manifest_sha256": control.get("smoke_manifest_sha256"),
        "smoke_experiment_fingerprint": control.get(
            "smoke_experiment_fingerprint"
        ),
    }
    if actual != expected:
        mismatches = {
            key: (actual[key], expected[key])
            for key in expected
            if actual[key] != expected[key]
        }
        raise ValueError(f"Selected protocol config mismatch: {mismatches}")
    return value
