# CI and contributions

- Read affected package guidance before changing checks.
- Keep PR, main-push, merge-group, manual-dispatch, reusable-workflow events. `github.base_ref` is not always set outside PRs.
- Change detection must include workflows and relevant shared actions/config. Test-only and tooling-only changes must trigger owning checks.
- Keep tool versions, lockfiles, reproducible installs, hardware/platform matrices, pinned action revisions.
- Prefer package completion commands over duplicating them in YAML. Webv2 `pnpm check:release`: install Chromium and required Linux dependencies first.
- Keep failure diagnostics with `always()` where fitting. Generated review/performance artifacts stay out of Git.
- Hooks must match CI's Ruff version and config. Validation never rewrites formatting.
- Follow [the PR template](pull_request_template.md). Short summaries. Omit subagent counts, tiers, process narration.
- Workflow changes neither configure hosted branch protection nor install local hooks. Verify before claiming either.
