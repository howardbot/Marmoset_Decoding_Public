"""Audit result-path conventions after the Results tree migration."""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "Code"
RESULTS_DIR = REPO_ROOT / "Results"
FORBIDDEN_PATTERNS = {
    "legacy generalization path": re.compile(r"Results/generalization"),
    "legacy manifold path": re.compile(r"Results/manifold_geometry"),
    "legacy archive path": re.compile(r"Results/legacy"),
    "legacy generalization components": re.compile(
        r'["\']Results["\']\s*/\s*["\']generalization["\']'
    ),
    "legacy manifold components": re.compile(
        r'["\']Results["\']\s*/\s*["\']manifold_geometry["\']'
    ),
}
ALLOWED_ROOT_FILES = {"README.md"}


def main() -> int:
    """Report stale code paths or loose files in the Results root."""
    violations: list[str] = []
    for path in sorted(CODE_DIR.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{label}: {path.relative_to(REPO_ROOT)}:{line}"
                )

    loose = sorted(
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_file() and path.name not in ALLOWED_ROOT_FILES
    )
    violations.extend(
        f"loose Results-root artifact: {path.relative_to(REPO_ROOT)}"
        for path in loose
    )

    if violations:
        print("\n".join(f"VIOLATION {item}" for item in violations))
        print(f"{len(violations)} result-path violation(s)")
        return 1
    print("Result paths follow current/workflows/archive conventions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
