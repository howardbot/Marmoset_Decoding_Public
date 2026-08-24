"""Small command router for the actively maintained analyses.

This keeps entry commands discoverable without hiding the underlying scripts.
Run ``python Code/generalization/run_analysis.py list`` for the available jobs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
JOBS = {
    "forget-equal-n": [
        PYTHON,
        "Code/generalization/analyses/forget_control_equal_n_crossday.py",
    ],
    "ty-locked-matrix": [
        PYTHON,
        "Code/generalization/build_locked_position_matrices.py",
        "--animal",
        "TY",
    ],
    "ty-significance": [
        PYTHON,
        "Code/generalization/analyses/position_asymmetry_significance.py",
    ],
    "data-manifest": [
        PYTHON,
        "Code/generalization/tools/build_data_manifest.py",
    ],
    "project-manifests": [
        PYTHON,
        "Code/generalization/tools/build_project_manifests.py",
    ],
    "check-links": [
        PYTHON,
        "Code/generalization/tools/check_markdown_links.py",
    ],
}


def main(argv: list[str] | None = None) -> int:
    """List or execute one maintained job, forwarding remaining arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("list", *JOBS))
    args, remainder = parser.parse_known_args(argv)
    if args.job == "list":
        for name, command in JOBS.items():
            print(f"{name:18} {' '.join(command[1:])}")
        return 0
    completed = subprocess.run(
        [*JOBS[args.job], *remainder],
        cwd=REPO_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
