#!/usr/bin/env python3
"""Rebuild notebooks/<name>.ipynb from scripts/<name>.py (scripts are the source of truth).

Uses jupytext with --update, so outputs already in a local notebook are kept. Outputs are stripped
from what git stores by the `stripout` clean filter, so regenerating never leaks them.
Run from anywhere: python3 tools/regen_notebooks.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUPYTEXT = ROOT / ".venv" / "bin" / "jupytext"
if not JUPYTEXT.exists():
    JUPYTEXT = "jupytext"

scripts = sorted(p for p in (ROOT / "scripts").glob("*.py") if not p.is_symlink())
if not scripts:
    sys.exit("no scripts found in scripts/")

failed = []
for script in scripts:
    nb = ROOT / "notebooks" / f"{script.stem}.ipynb"
    # "--set-kernel -" picks the kernelspec that matches the interpreter jupytext runs in
    cmd = [str(JUPYTEXT), "--to", "ipynb", "--update", "--set-kernel", "-", "-o", str(nb), str(script)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        failed.append(script.name)
        print(f"FAILED {script.name}\n{result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"ok  {nb.relative_to(ROOT)}")

stems = {s.stem for s in scripts}
for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
    if nb.stem not in stems:
        print(f"warning: {nb.relative_to(ROOT)} has no script in scripts/ and is not regenerated", file=sys.stderr)

sys.exit(1 if failed else 0)
