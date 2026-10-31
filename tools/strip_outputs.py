#!/usr/bin/env python3
"""Git clean filter for notebooks: drop every cell output so none is ever committed.

Reads a notebook on stdin and writes the stripped notebook to stdout. The working copy keeps its
outputs; only what git stores changes. Registered as filter `stripout` by tools/setup.sh.
"""
import json
import sys

raw = sys.stdin.read()
try:
    nb = json.loads(raw)
except ValueError:
    sys.stdout.write(raw)  # not JSON: pass it through untouched
    sys.exit(0)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cell.get("metadata", {}).pop("execution", None)
nb.get("metadata", {}).pop("widgets", None)

sys.stdout.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
