#!/usr/bin/env bash
# One-time setup per clone: strip notebook outputs on commit and run the pre-push check.
# These are local git settings (.git/config), so every fresh clone needs to run this once.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

git config filter.stripout.clean "python3 tools/strip_outputs.py"
git config filter.stripout.smudge cat
git config filter.stripout.required true
git config core.hooksPath tools/hooks
chmod +x tools/hooks/pre-push tools/strip_outputs.py tools/regen_notebooks.py

echo "ok: notebook outputs are stripped on commit, and tools/hooks/pre-push runs before every push."
