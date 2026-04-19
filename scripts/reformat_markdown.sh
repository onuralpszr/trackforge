#!/usr/bin/env bash
# One-shot helper: reformat all markdown/yaml/json via prettier and
# apply markdownlint-cli2 autofix. Run this ONCE after installing prek,
# review the diff, then commit manually.
#
# Requires: npx (Node.js). prek will fetch the hook envs on first run.

set -euo pipefail

if ! command -v prek >/dev/null 2>&1; then
    echo "prek is not installed. Install via:"
    echo "  cargo binstall prek   # or: cargo install --locked prek"
    exit 1
fi

echo ">>> Running prek on all files (this installs hook environments on first run)"
prek run --all-files || true

echo
echo ">>> Done. Review with:"
echo "    git diff"
echo
echo "Then stage and commit selectively, e.g.:"
echo "    bash scripts/commit_plan.sh all"
