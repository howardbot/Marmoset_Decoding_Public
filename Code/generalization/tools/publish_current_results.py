"""Publish report-facing result artifacts from local workflow outputs.

Analysis programs write complete outputs below ``Results/workflows``.  This
tool copies the small, report-facing subset named in
``Results/current/source_map.csv`` into ``Results/current`` and verifies every
copy by SHA-256.  The explicit map keeps report snapshots reproducible without
tracking the full local results workspace in the public repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from build_project_manifests import write_result_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
CURRENT_RESULTS_DIR = REPO_ROOT / "Results" / "current"
SOURCE_MAP = CURRENT_RESULTS_DIR / "source_map.csv"


@dataclass(frozen=True)
class Publication:
    """One workflow artifact and its canonical report-facing destination."""

    source: Path
    current: Path


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def within(path: Path, parent: Path) -> bool:
    """Return whether a resolved path is inside the resolved parent directory."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def load_publications(mapping_path: Path = SOURCE_MAP) -> list[Publication]:
    """Load and validate the explicit workflow-to-current publication map."""
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"publication map is empty: {mapping_path}")

    publications = []
    seen_sources: set[Path] = set()
    seen_destinations: set[Path] = set()
    for row in rows:
        source = REPO_ROOT / row["source_path"]
        current = REPO_ROOT / row["current_path"]
        if not within(source, REPO_ROOT / "Results" / "workflows"):
            raise ValueError(f"source is outside Results/workflows: {source}")
        if not within(current, CURRENT_RESULTS_DIR):
            raise ValueError(f"destination is outside Results/current: {current}")
        if source in seen_sources:
            raise ValueError(f"duplicate source in publication map: {source}")
        if current in seen_destinations:
            raise ValueError(f"duplicate destination in publication map: {current}")
        seen_sources.add(source)
        seen_destinations.add(current)
        publications.append(Publication(source=source, current=current))
    return publications


def publish(publications: list[Publication]) -> None:
    """Copy mapped artifacts and verify source and destination content hashes."""
    missing = [item.source for item in publications if not item.source.is_file()]
    if missing:
        lines = "\n".join(f"  - {path.relative_to(REPO_ROOT)}" for path in missing)
        raise FileNotFoundError(f"missing workflow artifact(s):\n{lines}")

    for item in publications:
        item.current.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, item.current)
        if sha256(item.source) != sha256(item.current):
            raise OSError(f"publication hash mismatch: {item.current}")


def check(publications: list[Publication]) -> list[Publication]:
    """Return mapped artifacts that are missing or differ from their sources."""
    stale = []
    for item in publications:
        if not item.source.is_file() or not item.current.is_file():
            stale.append(item)
        elif sha256(item.source) != sha256(item.current):
            stale.append(item)
    return stale


def main(argv: list[str] | None = None) -> int:
    """Publish mapped results, or only check whether current is synchronized."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the current snapshot without copying any files",
    )
    args = parser.parse_args(argv)
    publications = load_publications()

    if args.check:
        stale = check(publications)
        if stale:
            for item in stale:
                print(
                    "STALE "
                    f"{item.source.relative_to(REPO_ROOT)} -> "
                    f"{item.current.relative_to(REPO_ROOT)}"
                )
            print(f"{len(stale)} of {len(publications)} publication(s) are stale")
            return 1
        print(f"All {len(publications)} current publications match workflow sources")
        return 0

    publish(publications)
    result_count = write_result_manifest()
    print(f"Published and hash-verified {len(publications)} result artifacts")
    print(f"Indexed {result_count} current files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
