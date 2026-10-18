# Working on InvokeAI

## Communication

- Updates: plain, short, outcome first. Name blockers and evidence. Separate verified results, assumptions, unavailable checks.
- Ask only when repo or request cannot settle a decision safely. Continue authorized work without re-asking.

## Find the right owner

Before editing, read every `AGENTS.md` along each target path. Scoped rules add to this file. Policy lives in `AGENTS.md`.

| Area | Guidance and source of truth |
| --- | --- |
| Active frontend | [webv2/AGENTS.md](invokeai/frontend/webv2/AGENTS.md), its `ARCHITECTURE.md`, executable architecture policy |
| API, services, invocations | [app/AGENTS.md](invokeai/app/AGENTS.md) |
| Inference, model management | [backend/AGENTS.md](invokeai/backend/AGENTS.md) |
| Python tests | [tests/AGENTS.md](tests/AGENTS.md), plus guidance for the code under test |
| Legacy frontend | [webv1/AGENTS.md](invokeai/frontend/webv1/AGENTS.md) |
| Shared generated API contracts | [api/AGENTS.md](invokeai/frontend/api/AGENTS.md) |
| CI, contribution tooling | [.github/AGENTS.md](.github/AGENTS.md) |
| Documentation | `docs/README.md`, `docs/package.json`. Keep product docs current. |

Frontend work targets **webv2**, served by default. `--webv2` is a compatibility alias; `--web-legacy` selects legacy web. UI investigations: verify build and launch target first. `frontend/api` owns generated OpenAPI/type artifacts.

## Engineering standard

- Before designing: read implementations, callers, contracts, tests. Define observable outcome and validation. Reproduce bugs where feasible. Fix root causes. Rethink designs that pile up patches.
- Keep ownership and interfaces clear. Cover edge cases, failures, data lifecycles, operational limits.
- Every function, dependency, fallback, test earns its place. Abstractions must translate contracts, isolate dependencies, own lifecycles, prevent drift, or simplify callers. No pass-through wrappers, speculative extensibility, redundant validation/config, or abstraction justified only by similar-looking code.
- Remove obsolete code, tests, scaffolding. Preserve unrelated work. Comment only non-obvious intent, constraints, trade-offs.
- Test meaningful behavior through its owning interface. Expected values independent; tests catch plausible regressions. No trivial constant/type/re-export assertions, incidental snapshots, duplicate coverage, mock choreography. Prefer real lightweight dependencies and isolated fixtures. Use doubles for expensive or external systems. Optimize test value, not count.
- Dependency and lockfile changes only on purpose. Never weaken assertions, gates, architecture policy, or performance budgets to pass. Explain existing failures and verification limits.
- Keep architecture and reference docs matching the code. State current invariants, not migration history. Write paths from a stated root. Give the reason for a non-obvious rule.
- Repo guidance beats external skill, plugin, workflow defaults: test boundaries follow existing owner interfaces and tests; reviews use the tiers below over the full task-owned diff, uncommitted files included; notes go in `.scratch/agents/<task>/`; commands use package-owned scripts and pnpm. No personal paths, no generic ceremony.

## Performance and efficiency

- Check affected hot paths and immediate callers for repeated work, renders/subscriptions, waterfalls, queries, serialization/copies, unbounded collections, retained resources.
- Do clear, low-risk improvements in scope. Larger optimizations must serve the task. Note unrelated opportunities briefly; do not expand scope.
- Measure nontrivial optimizations and performance claims: existing budgets, representative fixtures, query counts, profiling, or focused benchmarks. Report material before/after and limits.
- Correctness and clarity first. Caching, memoization, concurrency, batching need understood invalidation, lifetimes, ordering, cost. No invented optimizations or benchmarks for a checklist.

## Product quality

Polished desktop experience: deliberate layouts, responsive interactions, cohesive accessible controls, complete loading/empty/error/recovery states. Verify changed interactions in the browser. Automated checks alone do not prove visual quality.

## Milestones and code review rules

Milestone = one coherent, commit-ready unit. Run relevant checks and self-review every change. Pick review depth by the whole task's behavior, blast radius, failure consequences. Not by file/line count or directory alone. Never split risky work to dodge review.

| Review depth | When |
| --- | --- |
| **Self-review; no independent reviewer** | Local, low-risk, easy to verify: copy, docs, styling, input minimum without shared validation, persistence, or API contract changes |
| **One independent reviewer** | Substantive feature, fix, refactor; bounded impact; no high-risk trigger |
| **Three independent reviewers** | Behavior changes touching authentication/authorization, account isolation, persisted-data compatibility/migrations, destructive operations, concurrency/resource lifecycles, shared API compatibility, inference numerics/device/memory, or substantial cross-owner architecture |

Review focuses: (1) correctness, spec conformance; (2) architecture, operational safety, performance, needless complexity; (3) test value and gaps, product quality, UI accessibility. One reviewer covers all three; three reviewers split them. User instructions override defaults. Pick the tier without asking; say briefly why you spawn reviewers. Required checks and browser verification apply at every tier.

1. Review once per coherent candidate, after checks and after every writer finishes. No edits to a candidate under review. Reviewers did not implement it. Give all reviewers the same fixed base, acceptance criteria, full task-owned diff (staged, unstaged, new files). Name unrelated changes. Focused briefs and entry points, not full conversation history.
2. Reviewers inspect independently, including needed callers/contracts. Read-only: no edits, commits, or delegation. Findings need location, impact, and failure scenario or engineering cost. Style preference alone is not a blocker.
3. Resolve all material findings. Add useful regression coverage. Rerun affected checks. Clean review is final for an unchanged candidate. Later edits: list changes since the reviewed candidate; get blocker-only follow-up from affected reviewers on them and their consequences. Broaden review or change tier only when scope/risk changes materially.
4. Never declare ready with material findings or unexplained failing gates. Report resolved findings, passed/failed/unavailable checks, remaining limits. PRs omit subagent counts, tiers, process narration.

Required independent review unavailable: self-review the matching focuses and tell the user. Ordinary low-risk self-review needs no disclaimer. Self-review is never independent, including review by an implementing agent.

## Delegated implementation

Main agent never edits repository files: every edit goes to an implementation subagent, so the main conversation keeps its context for planning, coordination, and review. Harness without subagents: edit locally.

- Main agent investigates, plans, briefs, integrates, validates, reports. It owns shared contracts and completion.
- Brief each implementer: objective, owned files, applicable `AGENTS.md`, agreed interfaces, acceptance criteria, focused checks. Implementer returns a short report: files changed, checks run, open issues.
- One writer per file at a time. Tightly coupled changes go to one implementer; split only independently verifiable work.
- Implementers own meaningful tests for their changes; by default never split production code from its tests.
- Follow-up fixes from review findings or failing checks go back to an implementer, preferably the original one.

Main agent also owns machine-wide resources: dependency installs, generated API contracts, full suites and release gates, GPU/generation work, fixed-port servers, shared browser sessions. Implementers run focused checks only, use spare ports, clean up temporary output, and never stash, reset, or check out in shared worktrees. Pre-commit hook stashes unstaged changes while it runs: commit only when no other agent writes to the same worktree. One webv2 performance gate per machine: it measures timing and binds fixed ports.

## Commands and environments

- **pnpm 10** only, never npm/yarn. Package-owned lockfiles. Run pnpm from the package directory: Corepack picks pnpm's version from the current directory, so `pnpm -C <package>` from the root can get the wrong one. Node version from `.nvmrc`. Read package scripts; do not assume the root Makefile covers webv2.
- Keep the existing Python/accelerator environment and `uv.lock`. Fresh test setup: `uv sync --locked --extra test`. Do not routinely replace accelerator backends.
- Root Python checks: `uv tool run ruff@0.11.2 check <paths>`, `uv tool run ruff@0.11.2 format --check <paths>`. 120 columns, absolute imports, configured vendored-code exclusions.
- Focused tests: `uv run --no-sync pytest <test-paths>`. Full CI: `uv run --no-sync pytest -n logical`. Hardware/quiet-machine timing tests: `-m slow` on dev machines; excluded by default and in CI. Isolation rules: `tests/AGENTS.md`.
- `-n logical` fits CI runners that own the machine. Dev box: cap workers (`-n 4`, or plain `pytest`). Set `OMP_NUM_THREADS` when a run must stay light. Never start a suite during a generation.
- `uv run --no-sync mypy scripts/invokeai-web.py` uses current exclusions. Optional diagnostic only: not a CI gate, not full type coverage.
- Pick checks by behavior. Run required milestone gates before review. Rerun unchanged checks only with a reason.
- Once test dependencies exist, install hooks per checkout: `uv run --no-sync pre-commit install`. Committed config does not install hooks.

## Scratch files and repository hygiene

- Never commit plans, work logs, investigation notes, review transcripts, handoffs, under any filename. Optional scratch: `.scratch/agents/<task>/`. Never relabel scratch as durable docs.
- Version durable docs of implemented architecture, decisions, APIs, setup.
- Check staged diffs and new files for planning material. Ignore rules neither untrack files nor block force-adds. Never force-add planning artifacts. Never add `package-lock.json`/`yarn.lock`.

## Commits and pull requests

- Readiness does not authorize commits or pushes. Follow the user's existing authorization. No automatic per-subtask commits.
- Conventional Commits. Subject: short, outcome first, scoped by feature area (`fix(queue)`). Never write the frontend package name `webv2` anywhere in the message. Body: one to three factual lines of material changes; no repetition, no process narration.
- Use the configured Git author.
- Follow `.github/pull_request_template.md`: headings in order, every applicable section and its inline guidance. Omit only conditional sections. Tick only true checklist items.
