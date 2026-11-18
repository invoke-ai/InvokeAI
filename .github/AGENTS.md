# CI and contributions

- Read affected package guidance before changing checks.
- Keep PR, main-push, merge-group, manual-dispatch, reusable-workflow events. `github.base_ref` is not always set outside PRs.
- Change detection must include workflows and relevant shared actions/config. Test-only and tooling-only changes must trigger owning checks.
- Keep tool versions, lockfiles, reproducible installs, hardware/platform matrices, pinned action revisions.
- Prefer package completion commands over duplicating them in YAML. Webv2 `pnpm check:release`: install locked Chromium, Firefox, WebKit and required Linux dependencies first. Completion includes the three-engine lifecycle smoke and reuses its performance build.
- Keep failure diagnostics with `always()` where fitting. Generated review/performance artifacts stay out of Git.
- Hooks must match CI's Ruff version and config. Validation never rewrites formatting.
- Workflow changes neither configure hosted branch protection nor install local hooks. Verify before claiming either.
- Windows ARM64 lock gate (`required-environments`, `scripts/check_platform_locks.py`) blocks by design. When a bump trips it: pin the last version with a win_arm64 wheel behind an ARM64 marker, drop an optional package there by marker with a degraded path and a `system-requirements.mdx` entry, or self-build it (procedure in `build-win-arm64-wheels.yml`'s header).

## Commits and pull requests

- Conventional Commits. Subject: short, outcome first, scoped by feature area (`fix(queue)`). Never write the frontend package name `webv2` anywhere in the message. Body: one to three factual lines of material changes; no repetition, no process narration.
- Use the configured Git author.
- The pre-commit hook stashes unstaged changes while it runs: commit only when no other agent writes to the same worktree.
- Follow [the PR template](pull_request_template.md): headings in order, every applicable section and its inline guidance. Omit only conditional sections. Tick only true checklist items. Short summaries. Omit subagent counts, tiers, process narration.
