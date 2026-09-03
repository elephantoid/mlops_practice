"""Fail the commit when an index document points at something that is not there.

This repo's recurring documentation failure is not typos, it is references that were
true once, or aspirational, or copied from a template -- `CONTEXT.md`, `docs/adr/`,
`../blueprint/track-e2e/`, `docs/agents/domain.md`. Each one sent a reader somewhere
empty and none of them announced itself.

Scope is deliberately narrow. Only the *index* documents are checked -- the ones that
describe what is true now:

    CLAUDE.md        STATUS.md        .claude/rules/*.md

`AGENTS.md` is exempt on purpose. It is the plan, so it names files that do not exist
yet (`.github/workflows/deploy.yml`) and one that lives in another repository
entirely. Holding a plan to "everything you mention must exist today" would make it
unwritable.

A reference passes if the path exists, or if git ignores it. The second clause matters:
`data/raw/`, `mlruns/` and `build/` are absent from every clone by design, and naming
them is correct rather than broken.

Only backtick-quoted tokens that look like paths are considered -- something ending in a
known file extension, or in a slash. That deliberately skips `Report.run`,
`evidently.legacy` and `models:/churnwatch@production`, which are code, not paths.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EXTENSIONS = {
    ".cfg",
    ".csv",
    ".db",
    ".example",
    ".ini",
    ".ipynb",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# Backticked runs of path-safe characters only. Anything with a space, quote, bracket,
# colon or "@" is code or prose and never reaches the existence check.
CANDIDATE = re.compile(r"`([A-Za-z0-9_./-]+)`")


def looks_like_a_path(token: str) -> bool:
    if token.endswith("/"):
        return True
    return Path(token).suffix in EXTENSIONS


def git_ignored(paths: list[str]) -> set[str]:
    """Paths git is configured to ignore. Absent by design, so not a broken reference."""
    if not paths:
        return set()
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def main(argv: list[str]) -> int:
    findings: list[tuple[str, int, str]] = []
    unresolved: list[tuple[str, int, str]] = []

    for filename in argv:
        for lineno, line in enumerate(Path(filename).read_text().splitlines(), start=1):
            for token in CANDIDATE.findall(line):
                if not looks_like_a_path(token) or Path(token).exists():
                    continue
                unresolved.append((filename, lineno, token))

    ignored = git_ignored([token for _, _, token in unresolved])
    findings = [entry for entry in unresolved if entry[2] not in ignored]

    for filename, lineno, token in findings:
        print(f"{filename}:{lineno}: references `{token}`, which does not exist")

    if findings:
        print(
            "\nAn index document must not point at something that is not there. "
            "Fix the path, create the file, or move the claim to AGENTS.md if it "
            "describes the plan rather than the present."
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
