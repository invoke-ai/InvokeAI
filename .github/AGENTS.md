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
- Windows ARM64 lock gate (`required-environments`, `scripts/check_platform_locks.py`) blocks by design. When a bump trips it: pin the last version with a win_arm64 wheel behind an ARM64 marker, drop an optional package there by marker with a degraded path and a `system-requirements.mdx` entry, or self-build it (procedure in `build-win-arm64-wheels.yml`'s header).
