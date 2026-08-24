"""Check local links in the canonical project indexes and Markdown reports."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[3]
LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CANONICAL_INPUTS = (
    REPO_ROOT / "PROJECT_INDEX.md",
    REPO_ROOT / "Data" / "README.md",
    REPO_ROOT / "Results" / "README.md",
    REPO_ROOT / "Results" / "current" / "README.md",
    REPO_ROOT / "Reports" / "README.md",
    REPO_ROOT / "References" / "README.md",
    REPO_ROOT / "Code" / "generalization" / "README.md",
    REPO_ROOT / "Code" / "generalization" / "docs" / "README.md",
)


def markdown_files(check_all: bool) -> list[Path]:
    """Return either canonical entry points or every maintained Markdown file."""
    if not check_all:
        files = [path for path in CANONICAL_INPUTS if path.exists()]
        files.extend((REPO_ROOT / "Reports" / "current").rglob("*.md"))
        files.extend((REPO_ROOT / "Reports" / "supporting").rglob("*.md"))
        return sorted(set(files))
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if not any(part in {".git", "tmp"} for part in path.parts)
    )


def local_target(source: Path, raw_target: str) -> Path | None:
    """Resolve a Markdown link target, returning ``None`` for external links."""
    target = raw_target.strip().split("#", 1)[0]
    if not target or target.startswith(("http://", "https://", "mailto:", "data:")):
        return None
    target = unquote(target)
    if target.startswith("/"):
        return Path(target)
    return (source.parent / target).resolve()


def main(argv: list[str] | None = None) -> int:
    """Print broken local links and return nonzero when any are found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="also check legacy notes")
    args = parser.parse_args(argv)
    broken = []
    files = markdown_files(args.all)
    for source in files:
        text = source.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = local_target(source, raw_target)
            if target is not None and not target.exists():
                broken.append((source, raw_target))
    if broken:
        for source, target in broken:
            print(f"BROKEN {source.relative_to(REPO_ROOT)} -> {target}")
        print(f"{len(broken)} broken local link(s) across {len(files)} Markdown files")
        return 1
    print(f"All local links valid across {len(files)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
