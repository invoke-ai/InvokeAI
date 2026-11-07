# Working on InvokeAI

## Communication

- Updates: plain, short, outcome first. Name blockers and evidence. Separate verified results, assumptions, unavailable checks.
- Ask only when repo or request cannot settle a decision safely. Continue authorized work without re-asking.

## Find the right owner

Before editing, read every `AGENTS.md` along each target path. Scoped rules add to this file. Policy lives in `AGENTS.md` files and the `.agents/` file they link.

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

Before these steps, read:

- Delegating work or running a milestone review: [.agents/coordination.md](.agents/coordination.md).
- Running Python tests: [tests/AGENTS.md](tests/AGENTS.md).
- Committing or opening a PR: [.github/AGENTS.md](.github/AGENTS.md).

## Engineering standard

- Before designing: read implementations, callers, contracts, tests. Define observable outcome and validation. Reproduce bugs where feasible. Fix root causes. Rethink designs that pile up patches.
- Keep ownership and interfaces clear. Cover edge cases, failures, data lifecycles, operational limits.
- Every function, dependency, fallback, test earns its place. Abstractions must translate contracts, isolate dependencies, own lifecycles, prevent drift, or simplify callers. No pass-through wrappers, speculative extensibility, redundant validation/config, or abstraction justified only by similar-looking code.
- Remove obsolete code, tests, scaffolding. Preserve unrelated work. Comment only non-obvious intent, constraints, trade-offs.
- Test meaningful behavior through its owning interface. Expected values independent; tests catch plausible regressions. No trivial constant/type/re-export assertions, incidental snapshots, duplicate coverage, mock choreography. Prefer real lightweight dependencies and isolated fixtures. Use doubles for expensive or external systems. Optimize test value, not count.
- Dependency and lockfile changes only on purpose. Never weaken assertions, gates, architecture policy, or performance budgets to pass. Explain existing failures and verification limits.
- Keep architecture and reference docs matching the code. State current invariants, not migration history. Write paths from a stated root. Give the reason for a non-obvious rule.
- Repo guidance beats external skill, plugin, workflow defaults: test boundaries follow existing owner interfaces and tests; reviews use the tiers below over the full task-owned diff, uncommitted files included; notes go in `.scratch/agents/<task>/`; commands use package-owned scripts and pnpm. No personal paths, no generic ceremony.
- Any webv2 UI change: verify changed interactions in the browser. Automated checks alone do not prove visual quality.

## Performance

- Check affected hot paths and immediate callers for repeated work, renders/subscriptions, waterfalls, queries, serialization/copies, unbounded collections, retained resources. Do clear, low-risk improvements in scope; larger ones only when the task needs them. Note unrelated opportunities briefly; do not expand scope.
- Correctness and clarity first. Measure nontrivial optimizations and performance claims (existing budgets, representative fixtures, query counts, profiling, focused benchmarks); report material before/after and limits. Caching, memoization, concurrency, batching need understood invalidation, lifetimes, ordering, cost. No invented optimizations or benchmarks for a checklist.

## Delegation

Main agent may edit directly only when the whole task qualifies for self-review; otherwise delegate implementation. Harness without subagents: edit locally.

Main agent owns machine-wide resources: dependency installs, generated API contracts, full suites and release gates, GPU/generation work, fixed-port servers, shared browser sessions. One webv2 performance gate per machine: it measures timing and binds fixed ports. Implementers run focused checks only, use spare ports, clean up temporary output, and never stash, reset, or check out in shared worktrees.

## Review and readiness

Milestone = one coherent, commit-ready unit. Run relevant checks and self-review every change. Pick review depth by the whole task's behavior, blast radius, failure consequences. Not by file/line count or directory alone. Never split risky work to dodge review.

| Review depth | When |
| --- | --- |
| **Self-review; no independent reviewer** | Local, low-risk, easy to verify: copy, docs, styling, input minimum without shared validation, persistence, or API contract changes |
| **One independent reviewer** | Substantive feature, fix, refactor; bounded impact; no high-risk trigger |
| **Three independent reviewers** | Behavior changes touching authentication/authorization, account isolation, persisted-data compatibility/migrations, destructive operations, concurrency/resource lifecycles, shared API compatibility, inference numerics/device/memory, or substantial cross-owner architecture |

- Reviewers did not implement the change. Self-review is never independent, including review by an implementing agent. Required independent review unavailable: self-review the matching focuses and tell the user. Ordinary low-risk self-review needs no disclaimer.
- Never declare ready with material findings or unexplained failing gates. Report resolved findings, passed/failed/unavailable checks, remaining limits. PRs omit subagent counts, tiers, process narration.
- Readiness does not authorize commits or pushes. Follow the user's existing authorization. No automatic per-subtask commits.

## Commands and environments

- **pnpm 10** only, never npm/yarn. Package-owned lockfiles. Run pnpm from the package directory: Corepack picks pnpm's version from the current directory, so `pnpm -C <package>` from the root can get the wrong one. Node version from `.nvmrc`. Read package scripts; do not assume the root Makefile covers webv2.
- Keep the existing Python/accelerator environment and `uv.lock`. Fresh test setup: `uv sync --locked --extra test`. Do not routinely replace accelerator backends.
- Once test dependencies exist, install hooks per checkout: `uv run --no-sync pre-commit install`. Committed config does not install hooks.
- Root Python checks: `uv tool run ruff@0.11.2 check <paths>`, `uv tool run ruff@0.11.2 format --check <paths>`. 120 columns, absolute imports, configured vendored-code exclusions.
- Focused tests: `uv run --no-sync pytest <test-paths>`. Never start a suite during a generation.
- Pick checks by behavior. Run required milestone gates before review. Rerun unchanged checks only with a reason.
- Family E2E runs (`scripts/family_e2e`) after changes to what a model computes or how it loads: model-manager loading, probing, configs; quantization; attention; model cache, working memory; invocations that run models; webv2 graph builders. None for docs-, test- or UI-only changes. Offer a run of the affected families, never start one unasked. Offer only where a manifest is set up; otherwise name the missing run under open limits in the final report. Procedure: `docs/src/content/docs/development/Guides/family-e2e.mdx`.

## Scratch files and repository hygiene

- Never commit plans, work logs, investigation notes, review transcripts, handoffs, under any filename. Optional scratch: `.scratch/agents/<task>/`. Never relabel scratch as durable docs.
- Version durable docs of implemented architecture, decisions, APIs, setup.
- Check staged diffs and new files for planning material. Ignore rules neither untrack files nor block force-adds. Never force-add planning artifacts. Never add `package-lock.json`/`yarn.lock`.
