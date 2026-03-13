#!/usr/bin/env python3
"""Validate exported ONNX metadata against dataset labels."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail export validation when ONNX embedded class names diverge from dataset labels.",
    )
    parser.add_argument("onnx_path", help="Path to the exported ONNX artifact")
    parser.add_argument(
        "--label-map",
        required=True,
        help="Path to a dataset label map (.pbtxt) describing the exported classes",
    )
    return parser.parse_args(argv)


def parse_label_map_pbtxt(label_map_path: str | Path) -> Dict[int, str]:
    """Parse a TensorFlow-style label map pbtxt into normalized zero-based indices."""
    text = Path(label_map_path).read_text(encoding="utf-8")
    entries: list[Tuple[int, str]] = []
    current_id: int | None = None
    current_name: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("id:"):
            current_id = int(_strip_pbtxt_trailing_comma(line.split(":", 1)[1].strip()))
        elif line.startswith("name:") or line.startswith("display_name:"):
            current_name = _strip_pbtxt_string(_strip_pbtxt_trailing_comma(line.split(":", 1)[1].strip()))
        elif line.startswith("}"):
            if current_id is not None and current_name is not None:
                entries.append((current_id, current_name))
            current_id = None
            current_name = None

    if current_id is not None and current_name is not None:
        entries.append((current_id, current_name))

    if not entries:
        raise ValueError(f"No label-map entries found in {label_map_path}")

    sorted_entries = sorted(entries, key=lambda item: item[0])
    minimum_id = sorted_entries[0][0]
    if minimum_id not in (0, 1):
        raise ValueError(
            f"Unsupported label-map base index {minimum_id} in {label_map_path}; expected 0- or 1-based ids"
        )

    offset = minimum_id
    normalized = {class_id - offset: name for class_id, name in sorted_entries}
    expected_indices = list(range(len(sorted_entries)))
    if sorted(normalized) != expected_indices:
        raise ValueError(
            f"Label-map ids in {label_map_path} are not contiguous after normalization: {sorted(normalized)}"
        )

    return normalized


def load_onnx_embedded_names(onnx_path: str | Path) -> Dict[int, str]:
    """Load embedded ONNX class names from custom metadata."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    for key in ("names", "class_names", "labels", "classes"):
        value = metadata.get(key)
        if not value:
            continue
        parsed = _parse_metadata_value(value)
        if parsed:
            return parsed

    raise ValueError(f"No embedded class-name metadata found in {onnx_path}")


def compare_name_maps(expected: Dict[int, str], actual: Dict[int, str]) -> list[str]:
    """Return mismatch lines between expected and actual name maps."""
    mismatches: list[str] = []
    expected_ids = set(expected)
    actual_ids = set(actual)

    for class_id in sorted(expected_ids | actual_ids):
        if class_id not in expected:
            mismatches.append(f"Unexpected ONNX class {class_id}: {actual[class_id]!r}")
            continue
        if class_id not in actual:
            mismatches.append(f"Missing ONNX class {class_id}: expected {expected[class_id]!r}")
            continue
        if expected[class_id].strip() != actual[class_id].strip():
            mismatches.append(
                f"Class {class_id} mismatch: expected {expected[class_id]!r}, got {actual[class_id]!r}"
            )

    return mismatches


def validate_export(onnx_path: str | Path, label_map_path: str | Path) -> int:
    expected_names = parse_label_map_pbtxt(label_map_path)
    embedded_names = load_onnx_embedded_names(onnx_path)
    mismatches = compare_name_maps(expected_names, embedded_names)

    print(f"Validated ONNX: {onnx_path}")
    print(f"Expected names: {expected_names}")
    print(f"Embedded names: {embedded_names}")

    if mismatches:
        print("ONNX export validation failed:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"  - {mismatch}", file=sys.stderr)
        return 1

    print("ONNX export metadata matches dataset labels.")
    return 0


def _parse_metadata_value(raw_value: str) -> Dict[int, str]:
    parsed = _coerce_structure(raw_value)
    if isinstance(parsed, list):
        return {index: str(value) for index, value in enumerate(parsed)}
    if isinstance(parsed, dict):
        normalized: Dict[int, str] = {}
        for key, value in parsed.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    return {}


def _coerce_structure(raw_value: str):
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(raw_value)
        except Exception:
            continue
    raise ValueError(f"Unsupported ONNX metadata value: {raw_value!r}")


def _strip_pbtxt_string(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _strip_pbtxt_trailing_comma(value: str) -> str:
    return value[:-1] if value.endswith(',') else value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return validate_export(args.onnx_path, args.label_map)
    except Exception as error:
        print(f"Validation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())