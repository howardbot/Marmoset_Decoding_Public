"""Build the lightweight manifest for processed NWB files in ``Data/``."""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path


GENERALIZATION_DIR = Path(__file__).resolve().parents[1]
if str(GENERALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERALIZATION_DIR))

from project_config import (  # noqa: E402
    DATA_DIR,
    TS_FORGET_R1,
    TS_FORGET_R2,
    TS_INTERFERENCE_R1,
    TS_INTERFERENCE_R2,
    TY_INTERFERENCE_R1,
    TY_INTERFERENCE_R2,
    session_date,
)


OUTPUT = DATA_DIR / "session_manifest.csv"
KNOWN_LIMITATIONS = {
    "TSAL20260609_0830_forgetAndForgetFree001": (
        "forget event 2 pose is all NaN in the current processed NWB; "
        "approximately 70 trials are unavailable pending reprocessing"
    ),
    "TSAL20260610_0830_forgetAndForgetFree001": (
        "complete processed file, but the animal performed only 31 usable "
        "S/F reaching trials"
    ),
}


def classify(session: str) -> tuple[str, str, str]:
    """Return animal, experiment, and epoch for a configured session."""
    grids = (
        ("TS", "interference", "R1", TS_INTERFERENCE_R1),
        ("TS", "interference", "R2", TS_INTERFERENCE_R2),
        ("TY", "interference", "R1", TY_INTERFERENCE_R1),
        ("TY", "interference", "R2", TY_INTERFERENCE_R2),
        ("TS", "forget_control", "R1", TS_FORGET_R1),
        ("TS", "forget_control", "R2", TS_FORGET_R2),
    )
    for animal, experiment, epoch, sessions in grids:
        if session in sessions:
            return animal, experiment, epoch
    animal = "TS" if session.startswith("TSAL") else "TY" if session.startswith("TYTR") else "unknown"
    return animal, "unregistered", "unknown"


def build_rows() -> list[dict[str, object]]:
    """Collect one manifest row per processed NWB without opening the file."""
    rows = []
    for path in sorted(DATA_DIR.glob("*_processed.nwb")):
        session = path.name.removesuffix("_processed.nwb")
        animal, experiment, epoch = classify(session)
        stat = path.stat()
        rows.append(
            {
                "session": session,
                "date": session_date(session),
                "animal": animal,
                "experiment": experiment,
                "epoch": epoch,
                "relative_path": path.relative_to(DATA_DIR.parent).as_posix(),
                "size_bytes": stat.st_size,
                "modified_local": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "known_limitation": KNOWN_LIMITATIONS.get(session, ""),
            }
        )
    return rows


def main() -> None:
    """Write ``Data/session_manifest.csv`` using stable column order."""
    rows = build_rows()
    columns = [
        "session",
        "date",
        "animal",
        "experiment",
        "epoch",
        "relative_path",
        "size_bytes",
        "modified_local",
        "known_limitation",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
