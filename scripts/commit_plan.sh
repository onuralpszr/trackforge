#!/usr/bin/env bash
# Stage and (optionally) commit changes in logical groups, each with a
# Signed-off-by trailer (git commit -sm).
#
# Usage:
#   bash scripts/commit_plan.sh [--dry-run|--stage-only|--commit] [group]
#
# Modes:
#   --commit       (default) stage the group and run `git commit -sm`
#   --stage-only   only `git add` the group; you commit yourself
#   --dry-run      print what would happen; touch nothing
#
# Groups:
#   prettier-md    prettier-reformatted markdown/yaml (README, docs, workflows, etc.)
#   prek           pre-commit / prek config
#   lint-configs   prettier + markdownlint configs
#   rust-tools     typos + taplo configs
#   ci             CI workflow changes + new pre-commit / autofix workflows
#   scripts        helper scripts
#   all            run every group in order
#
# Re-run-safe: each group only stages files that actually changed.

set -euo pipefail

MODE="--commit"
case "${1:-}" in
    --dry-run | --stage-only | --commit)
        MODE="$1"
        shift
        ;;
esac
GROUP="${1:-all}"

run() {
    local msg="$1"
    shift
    local files=()
    for f in "$@"; do
        [[ -e "$f" ]] && files+=("$f")
    done

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "  (skip) $msg — no matching files"
        return 0
    fi

    echo
    echo "=============================================================="
    echo "  Group: $msg"
    echo "  Files:"
    printf '    %s\n' "${files[@]}"
    echo "=============================================================="

    case "$MODE" in
        --dry-run)
            echo "  Would run: git add -- ${files[*]}"
            echo "  Would run: git commit -sm \"$msg\""
            ;;
        --stage-only)
            git add -- "${files[@]}"
            echo "  Staged. Review with: git diff --cached"
            echo "  Then commit with:   git commit -sm \"$msg\""
            ;;
        --commit)
            git add -- "${files[@]}"
            # Only commit if something actually got staged for these paths.
            if git diff --cached --quiet -- "${files[@]}"; then
                echo "  (skip commit) no staged changes for this group"
            else
                git commit -sm "$msg"
            fi
            ;;
    esac
}

group_prettier_md() {
    run "style: apply prettier to markdown and yaml files" \
        README.md \
        CONTRIBUTING.md \
        docs/examples.md \
        docs/index.md \
        examples/python/README.md \
        src/trackers/byte_track/README.md \
        src/trackers/deepsort/README.md \
        src/trackers/sort/README.md \
        .github/dependabot.yml \
        .github/workflows/codencov.yaml \
        .github/workflows/docs.yml \
        .github/workflows/security-audit.yml
}

group_prek() {
    run "chore(prek): add pre-commit / prek config" \
        .pre-commit-config.yaml
}

group_lint_configs() {
    run "chore(lint): add prettier + markdownlint configuration" \
        .prettierrc.yaml \
        .prettierignore \
        .markdownlint.yaml \
        .markdownlintignore
}

group_rust_tools() {
    run "chore(lint): add typos and taplo configuration" \
        typos.toml \
        taplo.toml
}

group_ci() {
    run "ci: add prek + autofix workflows; extend CI with deny/machete/typos/taplo/msrv" \
        .github/workflows/pre-commit.yml \
        .github/workflows/autofix.yml \
        .github/workflows/CI.yml
}

group_scripts() {
    run "chore(scripts): add commit-plan and markdown-reformat helpers" \
        scripts/commit_plan.sh \
        scripts/reformat_markdown.sh
}

case "$GROUP" in
    prettier-md)  group_prettier_md ;;
    prek)         group_prek ;;
    lint-configs) group_lint_configs ;;
    rust-tools)   group_rust_tools ;;
    ci)           group_ci ;;
    scripts)      group_scripts ;;
    all)
        group_prettier_md
        group_lint_configs
        group_rust_tools
        group_prek
        group_ci
        group_scripts
        ;;
    *)
        echo "Unknown group: $GROUP" >&2
        sed -n '1,30p' "$0" >&2
        exit 1
        ;;
esac

echo
case "$MODE" in
    --dry-run)
        echo "Dry run complete. Re-run without --dry-run to stage/commit."
        ;;
    --stage-only)
        echo "All groups staged. Inspect with: git diff --cached"
        echo "To unstage:  git reset HEAD"
        ;;
    --commit)
        echo "Done. Recent commits:"
        git log --oneline -n 10
        ;;
esac
