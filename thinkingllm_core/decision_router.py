"""Validation and small, dependency-light helpers for the decision router."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


MAX_DECISION_ROUTES = 52
OPENJEV_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def normalize_routes(values: Iterable[tuple[str, str]]) -> list[dict[str, object]]:
    """Validate route widget pairs and retain their fixed output-slot indexes."""
    routes: list[dict[str, object]] = []
    seen: dict[str, str] = {}

    for slot, pair in enumerate(values):
        name, description = pair
        name = str(name or "").strip()
        description = str(description or "").strip()
        if not name:
            if description:
                raise ValueError(f"Route {slot + 1} has a description but no route name.")
            continue
        if len(name) > 128:
            raise ValueError(f"Route {slot + 1} name is longer than 128 characters.")
        if any(ord(char) < 32 for char in name):
            raise ValueError(f"Route {slot + 1} name cannot contain line breaks or control characters.")
        if len(description) > 1024:
            raise ValueError(f"Description for route {name!r} is longer than 1024 characters.")

        duplicate_key = name.casefold()
        if duplicate_key in seen:
            raise ValueError(
                f"Route names must be unique (case-insensitive): {seen[duplicate_key]!r} and {name!r}."
            )
        seen[duplicate_key] = name
        routes.append({"slot": slot, "name": name, "description": description})

    if len(routes) < 2:
        raise ValueError("Configure at least two non-empty route names.")
    if len(routes) > MAX_DECISION_ROUTES:
        raise ValueError(f"The Decision Router supports at most {MAX_DECISION_ROUTES} routes.")
    return routes


def stable_softmax(scores: Iterable[float], temperature: float = 1.0) -> list[float]:
    """Normalize finite model scores without overflow or underflow."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("Decision temperature must be a finite number greater than zero.")
    values = [float(score) / float(temperature) for score in scores]
    if len(values) < 2:
        raise ValueError("A decision needs scores for at least two routes.")
    if not all(math.isfinite(score) for score in values):
        raise RuntimeError("The decision model returned non-finite route scores.")
    peak = max(values)
    weights = [math.exp(score - peak) for score in values]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("The decision model returned scores that could not be normalized.")
    return [weight / total for weight in weights]


def map_route_probabilities(routes: list[dict[str, object]], probabilities: Iterable[float]) -> dict[str, float]:
    probs = [float(value) for value in probabilities]
    if len(routes) != len(probs):
        raise ValueError("The number of model probabilities does not match the configured routes.")
    return {str(route["name"]): probs[index] for index, route in enumerate(routes)}


def route_indices(routes: list[dict[str, object]], chosen_names: Iterable[str]) -> dict[int, list[int]]:
    """Build stable output-slot to input-batch-index mapping."""
    slot_by_name = {str(route["name"]): int(route["slot"]) for route in routes}
    result = {int(route["slot"]): [] for route in routes}
    for image_index, name in enumerate(chosen_names):
        try:
            result[slot_by_name[name]].append(image_index)
        except KeyError as exc:
            raise RuntimeError(f"The decision model selected an unknown route: {name!r}.") from exc
    return result


def selected_route(routes: list[dict[str, object]], probabilities: Mapping[str, float]) -> str:
    if set(probabilities) != {str(route["name"]) for route in routes}:
        raise ValueError("The probability map does not match the configured routes.")
    return max(probabilities, key=probabilities.__getitem__)
