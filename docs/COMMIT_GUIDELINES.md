# Commit guidelines

Trackforge follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). If
you want interactive help picking the type and scope, use
[Commitizen](https://commitizen-tools.github.io/commitizen/) with `cz commit`.

## Subject line

- Keep the subject to one plain sentence under 72 characters, no trailing period.
- Use a tracker scope for tracker changes (for example `feat(deep_ocsort): ...`) or a shared
  scope such as `deps`, `docs`, `ci`, `build`, or `refactor`.

## Rules

- Sign off every commit: `git commit -sm "..."`.
- No co-author trailers and no "generated with" or AI attribution lines. Disclose AI
  assistance in the PR description instead (see CONTRIBUTING.md).
- No em dashes or other decorative characters in the message.
- If the PR has a single commit, use that commit subject as the PR title.
- Keep the PR description in plain prose, without section headers.

## Examples

```text
feat(deep_ocsort): add camera motion compensation
fix(botsort): keep low score detections out of stage one
docs: list all three OC-SORT mechanisms in the API reference
chore(deps): bump opencv from 0.101.1 to 0.102.0
```
