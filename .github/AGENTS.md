# CI and contributions

- Read affected package guidance before changing checks.
- Preserve PR, main-push, merge-group, manual-dispatch, and reusable-workflow events. `github.base_ref` is not always populated outside PRs.
- Change detection must include workflows and relevant shared actions/configuration; test-only and tooling-only changes must trigger owning checks.
- Preserve tool versions, lockfiles, reproducible installs, hardware/platform matrices, and pinned action revisions.
- Prefer package completion commands over duplicating them in YAML. For webv2's `pnpm check:release`, install Chromium and required Linux dependencies first.
- Preserve failure diagnostics with `always()` where appropriate; keep generated review/performance artifacts out of Git.
- Match hooks to CI's Ruff version/configuration; validation must not rewrite formatting.
- Follow [the PR template](pull_request_template.md); keep summaries short and omit subagent counts, tiers, and process narration.
- Workflow changes do not configure hosted branch protection or install local hooks; verify before claiming either.
