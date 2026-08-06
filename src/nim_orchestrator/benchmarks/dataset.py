"""Versioned benchmark dataset loading and validation.

Development cases (benchmark_cases_v1.yaml) are used while building; sealed
cases (benchmark_cases_v1_sealed.yaml) are evaluation-only and must never be
inspected or tuned against. Checksums guard against silent edits.
"""
import hashlib
from pathlib import Path

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"
DEV_DATASET = DEFAULT_CONFIG_DIR / "benchmark_cases_v1.yaml"
SEALED_DATASET = DEFAULT_CONFIG_DIR / "benchmark_cases_v1_sealed.yaml"

MIN_CASES_PER_CATEGORY = {"dev": 10, "sealed": 5}

VALID_CHECKS = {
    "arithmetic", "code", "debug", "factual", "architecture",
    "security", "compound", "ambiguous", "adversarial",
}


class DatasetError(ValueError):
    """Raised when a benchmark dataset is missing, tampered or malformed."""


def load_cases(path: Path | str, *, min_per_category: int = MIN_CASES_PER_CATEGORY["dev"]) -> list[dict]:
    """Load and validate a benchmark dataset file."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"benchmark dataset not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise DatasetError(f"{path.name}: top-level structure must be a mapping")
    if "version" not in raw or not raw.get("version"):
        raise DatasetError(f"{path.name}: missing 'version'")

    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise DatasetError(f"{path.name}: no cases")

    # checksum guards against silent edits
    expected = raw.get("checksum")
    if expected:
        actual = hashlib.sha256(yaml.safe_dump(cases, sort_keys=True).encode()).hexdigest()[:16]
        if actual != expected:
            raise DatasetError(f"{path.name}: checksum mismatch — dataset tampered")

    by_category: dict[str, int] = {}
    seen_ids: set[str] = set()
    for c in cases:
        if "id" not in c or "question" not in c or "category" not in c:
            raise DatasetError(f"{path.name}: case missing id/question/category: {c!r}")
        if c["id"] in seen_ids:
            raise DatasetError(f"{path.name}: duplicate case id '{c['id']}'")
        seen_ids.add(c["id"])
        if c.get("check") not in VALID_CHECKS:
            raise DatasetError(f"{path.name}: case '{c['id']}' has invalid check '{c.get('check')}'")
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1

    for cat, count in by_category.items():
        if count < min_per_category:
            raise DatasetError(f"{path.name}: category '{cat}' has {count} cases (< {min_per_category})")

    return cases


def load_dev() -> list[dict]:
    return load_cases(DEV_DATASET)


def load_sealed() -> list[dict]:
    return load_cases(SEALED_DATASET, min_per_category=MIN_CASES_PER_CATEGORY["sealed"])


def dev_and_sealed_disjoint() -> bool:
    dev_ids = {c["id"] for c in load_dev()}
    sealed_ids = {c["id"] for c in load_sealed()}
    return dev_ids.isdisjoint(sealed_ids)
