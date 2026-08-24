"""Index Python scripts and curated current result artifacts."""
from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERALIZATION_DIR = REPO_ROOT / "Code" / "generalization"
CURRENT_RESULTS_DIR = REPO_ROOT / "Results" / "current"
SCRIPT_OUTPUT = GENERALIZATION_DIR / "script_manifest.csv"
RESULT_OUTPUT = CURRENT_RESULTS_DIR / "result_manifest.csv"
MAINTAINED_ENTRYPOINTS = {
    "Code/generalization/run_analysis.py",
    "Code/generalization/big_sweep_phase1_withinday.py",
    "Code/generalization/big_sweep_phase2_crossday.py",
    "Code/generalization/build_locked_position_matrices.py",
    "Code/generalization/analyses/forget_control_equal_n_crossday.py",
    "Code/generalization/analyses/position_asymmetry_significance.py",
    "Code/generalization/analyses/plot_interference_forget_paired_directional.py",
    "Code/generalization/analyses/plot_ty_paired_directional_significance.py",
}


def script_role(path: Path) -> str:
    """Infer a compact role label from a script's directory and filename."""
    name = path.name
    parts = path.relative_to(GENERALIZATION_DIR).parts
    if "hypothesis_function_tests" in parts or name.startswith("test_"):
        return "test"
    if "tools" in parts:
        return "project_tool"
    if "plots" in parts or name.startswith("plot_"):
        return "plot"
    if name.startswith("summarize_") or name.endswith("_summary.py"):
        return "summary"
    if "diagnostics" in parts:
        return "diagnostic"
    if "sweeps" in parts or "sweep" in name:
        return "sweep"
    if "pipeline_v1" in parts:
        return "legacy_pipeline"
    if name in {"project_config.py", "plotting_common.py"}:
        return "shared_config"
    return "analysis"


def write_script_manifest() -> int:
    """Write one row for every Python file in the generalization tree."""
    rows = []
    for path in sorted(GENERALIZATION_DIR.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        rows.append(
            {
                "relative_path": relative,
                "directory": path.parent.relative_to(GENERALIZATION_DIR).as_posix(),
                "filename": path.name,
                "role": script_role(path),
                "maintained_entrypoint": relative in MAINTAINED_ENTRYPOINTS,
                "size_bytes": path.stat().st_size,
            }
        )
    with SCRIPT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def result_kind(path: Path) -> str:
    """Return a simple artifact type for a current result path."""
    suffix = path.suffix.lower()
    if suffix in {".png", ".svg", ".pdf"}:
        return "figure"
    if suffix in {".csv", ".tsv", ".parquet"}:
        return "table"
    if suffix == ".md":
        return "index"
    return "other"


def write_result_manifest() -> int:
    """Write one row for every curated file under ``Results/current``."""
    rows = []
    for path in sorted(CURRENT_RESULTS_DIR.rglob("*")):
        if not path.is_file() or path == RESULT_OUTPUT:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(REPO_ROOT).as_posix(),
                "kind": result_kind(path),
                "size_bytes": path.stat().st_size,
            }
        )
    CURRENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ("relative_path", "kind", "size_bytes")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    """Regenerate both project-maintenance manifests."""
    script_count = write_script_manifest()
    result_count = write_result_manifest()
    print(f"Indexed {script_count} Python files in {SCRIPT_OUTPUT}")
    print(f"Indexed {result_count} current result files in {RESULT_OUTPUT}")


if __name__ == "__main__":
    main()
