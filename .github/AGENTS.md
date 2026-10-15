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

## Windows ARM64 lock gate

`uv lock` requires a win_arm64 resolution (`required-environments`), and `scripts/check_platform_locks.py` fails when any package an install pulls in on Windows ARM64 has no win_arm64 wheel. The check blocks by design: as a warning it would let a lock that fails mid-install on every ARM64 machine reach a release. When a dependency bump trips either:

- Pin the last version that has a win_arm64 wheel behind `; sys_platform == 'win32' and platform_machine == 'ARM64'`, beside the unqualified requirement narrowed with the opposite marker.
- If the package is optional, drop it on Windows ARM64 by marker, make its import site degrade with a clear message there, and list the difference in `system-requirements.mdx`.
- Otherwise build it in `build-win-arm64-wheels.yml`. Dispatch with a **new** `release_tag` (published tags are pinned by hash and never rewritten), then point the `invoke-win-arm64-wheels` index URL and the docs' `--find-links` at that tag, bump the matching pins, and re-lock. Bumping an already self-built wheel works the same way.
