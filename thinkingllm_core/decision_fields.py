"""Validation and result mapping for typed Decision Fields outputs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


MAX_BOOLEAN_FIELDS = 8
MAX_INTEGER_FIELDS = 8
MAX_DECISION_FIELDS = MAX_BOOLEAN_FIELDS + MAX_INTEGER_FIELDS
MAX_FIELD_OPTIONS = 52


def _integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def normalize_fields(values: Iterable[tuple[str, str, str, float, int, int]]) -> list[dict[str, object]]:
    """Validate field widgets and retain each field's stable configuration index."""
    fields: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    type_counts = {"BOOLEAN": 0, "INT": 0}

    for slot, pair in enumerate(values):
        name, kind, question, threshold, minimum, maximum = pair
        name = str(name or "").strip()
        kind = str(kind or "BOOLEAN").strip().upper()
        question = str(question or "").strip()
        if not name:
            if question:
                raise ValueError(f"Field {slot + 1} has a question but no field name.")
            continue
        if len(name) > 128:
            raise ValueError(f"Field {slot + 1} name is longer than 128 characters.")
        if any(ord(char) < 32 for char in name):
            raise ValueError(f"Field {slot + 1} name cannot contain line breaks or control characters.")
        if not question:
            raise ValueError(f"Field {name!r} needs a question or description.")
        if len(question) > 1024:
            raise ValueError(f"Question for field {name!r} is longer than 1024 characters.")
        if kind not in {"BOOLEAN", "INT"}:
            raise ValueError(f"Unsupported field type for {name!r}: {kind!r}.")
        type_counts[kind] += 1
        group_limit = MAX_BOOLEAN_FIELDS if kind == "BOOLEAN" else MAX_INTEGER_FIELDS
        if type_counts[kind] > group_limit:
            raise ValueError(f"Decision Fields supports at most {group_limit} {kind} outputs.")

        duplicate_key = name.casefold()
        if duplicate_key in seen:
            raise ValueError(
                f"Field names must be unique (case-insensitive): {seen[duplicate_key]!r} and {name!r}."
            )
        seen[duplicate_key] = name

        field: dict[str, object] = {
            "slot": slot,
            "id": f"field_{slot + 1:02d}",
            "name": name,
            "type": kind,
            "question": question,
        }
        if kind == "BOOLEAN":
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Boolean threshold for {name!r} must be a number from 0 to 1.") from exc
            if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
                raise ValueError(f"Boolean threshold for {name!r} must be a finite number from 0 to 1.")
            field["threshold"] = threshold_value
            field["options"] = [
                {"value": "true", "description": f"Yes. The condition is satisfied: {question}"},
                {"value": "false", "description": f"No. The condition is not satisfied: {question}"},
            ]
        else:
            minimum_value = _integer(minimum, f"Minimum for {name!r}")
            maximum_value = _integer(maximum, f"Maximum for {name!r}")
            if minimum_value > maximum_value:
                raise ValueError(f"Minimum for {name!r} must not be greater than its maximum.")
            count = maximum_value - minimum_value + 1
            option_count = count + (1 if minimum_value >= 0 else 2)
            if option_count > MAX_FIELD_OPTIONS:
                raise ValueError(
                    f"Integer range for {name!r} needs {option_count} choices; the local decision head supports at most {MAX_FIELD_OPTIONS} choices."
                )
            options: list[dict[str, str]] = []
            if minimum_value < 0:
                options.append({"value": "<min", "description": f"The integer is less than {minimum_value}."})
            options.extend(
                {"value": str(value), "description": f"The integer is exactly {value}."}
                for value in range(minimum_value, maximum_value + 1)
            )
            options.append({"value": ">max", "description": f"The integer is greater than {maximum_value}."})
            field.update({"minimum": minimum_value, "maximum": maximum_value, "options": options})
        fields.append(field)

    if len(fields) > MAX_DECISION_FIELDS:
        raise ValueError(f"Decision Fields supports at most {MAX_DECISION_FIELDS} configured fields.")
    return fields


def field_routes(field: Mapping[str, object]) -> list[dict[str, str]]:
    """Adapt typed output choices to the existing native decision-head interface."""
    return [
        {"name": str(option["value"]), "description": str(option["description"])}
        for option in field["options"]
    ]


def normalize_field_result(
    field: Mapping[str, object],
    choice: str,
    probabilities: Mapping[str, float],
    confidence: float | None = None,
) -> dict[str, object]:
    """Validate a categorical answer and map it to a typed ComfyUI scalar."""
    option_values = {str(option["value"]) for option in field["options"]}
    if choice not in option_values:
        raise RuntimeError(f"Decision model selected an unknown value for field {field['name']!r}.")
    if set(probabilities) != option_values:
        raise RuntimeError(f"Decision probabilities do not match field {field['name']!r}.")

    normalized_probabilities: dict[str, float] = {}
    for option in field["options"]:
        value = str(option["value"])
        probability = probabilities[value]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise RuntimeError(f"Decision model returned an invalid probability for field {field['name']!r}.")
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Decision model returned an invalid probability for field {field['name']!r}.")
        normalized_probabilities[value] = probability

    result: dict[str, object] = {
        "id": str(field["id"]),
        "name": str(field["name"]),
        "type": str(field["type"]),
        "value": None,
        "choice": choice,
        "probabilities": normalized_probabilities,
    }
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise RuntimeError(f"Decision model returned invalid confidence for field {field['name']!r}.")
        confidence_value = float(confidence)
        if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
            raise RuntimeError(f"Decision model returned invalid confidence for field {field['name']!r}.")
        result["confidence"] = confidence_value

    if field["type"] == "BOOLEAN":
        result["value"] = normalized_probabilities["true"] >= float(field["threshold"])
    elif choice == "<min":
        result.update({"value": int(field["minimum"]), "range_status": "below_min"})
    elif choice == ">max":
        result.update({"value": int(field["maximum"]), "range_status": "above_max"})
    else:
        try:
            integer_value = int(choice)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Decision model returned a non-integer value for field {field['name']!r}.") from exc
        if str(integer_value) != choice or not int(field["minimum"]) <= integer_value <= int(field["maximum"]):
            raise RuntimeError(f"Decision model returned an out-of-range integer for field {field['name']!r}.")
        result["value"] = integer_value
        result["range_status"] = "exact"
    return result
