# Working on InvokeAI

## Communication

- Give plain, concise updates led by outcomes, blockers, and evidence. Distinguish verified results, assumptions, and unavailable checks.
- Ask only when a decision cannot be inferred safely from the repository or request; continue authorized work without repeated permission requests.

## Find the right owner

Before editing, read `AGENTS.md` along each target path; scoped rules supplement this file. Keep policy in `AGENTS.md`.

| Area | Guidance and source of truth |
| --- | --- |
| Active frontend | [webv2/AGENTS.md](invokeai/frontend/webv2/AGENTS.md), its `ARCHITECTURE.md` and executable architecture policy |
| API, services, invocations | [app/AGENTS.md](invokeai/app/AGENTS.md) |
| Inference and model management | [backend/AGENTS.md](invokeai/backend/AGENTS.md) |
| Python tests | [tests/AGENTS.md](tests/AGENTS.md); also read guidance for the production code under test |
| Legacy frontend and generated API artifacts | [web/AGENTS.md](invokeai/frontend/web/AGENTS.md) |
| CI and contribution tooling | [.github/AGENTS.md](.github/AGENTS.md) |
| Documentation | `docs/README.md` and `docs/package.json`; keep documentation about the implemented product current |

Ordinary frontend work targets **webv2**, served with `--webv2`; default launches select legacy web. Verify build/launch targets for UI investigations. Legacy web owns generated OpenAPI/type artifacts.

## Engineering standard

- Inspect implementations, callers, contracts, and tests before designing changes. Define observable outcomes and validation; reproduce bugs where feasible, fix root causes, and reconsider designs that accumulate patches.
- Keep ownership and interfaces clear. Account for edge cases, failures, data lifecycles, and operational limits.
- Every function, dependency, fallback, and test must earn its place. Add abstractions to translate contracts, isolate dependencies, own lifecycles, prevent drift, or simplify callers; avoid pass-through wrappers, speculative extensibility, redundant validation/configuration, and abstractions justified only by similar-looking code.
- Remove obsolete code, tests, and scaffolding; preserve unrelated work. Comment only non-obvious intent, constraints, or trade-offs.
- Test meaningful behavior through its owning interface with independent expectations that catch plausible regressions. Avoid trivial constant/type/re-export assertions, incidental snapshots, duplicate coverage, and mock choreography. Prefer real lightweight dependencies and isolated fixtures; use doubles for expensive/external systems. Optimize test value, not count.
- Keep dependency/lockfile changes intentional. Never weaken assertions, gates, architecture policy, or performance budgets to pass; explain existing failures and verification limits.
- Keep architecture/reference docs aligned with implementation. Skills are optional helpers: adapt them to this stack without importing personal paths or generic ceremony.

## Performance and efficiency

- Inspect affected hot paths and immediate callers for repeated work, renders/subscriptions, waterfalls, queries, serialization/copies, unbounded collections, and retained resources.
- Implement clear, low-risk improvements in scope; larger optimizations must serve the task. Briefly note unrelated opportunities without expanding scope.
- Measure nontrivial optimizations and performance claims with existing budgets and representative fixtures, query counts, profiling, or focused benchmarks. Report material before/after results and limits.
- Preserve correctness and clarity. Require understood invalidation, lifetimes, ordering, and cost for caching, memoization, concurrency, or batching. Do not invent optimizations or benchmarks for a checklist.

## Product quality

Maintain a polished desktop experience: deliberate layouts, responsive interactions, cohesive accessible controls, and complete loading/empty/error/recovery states. Verify changed interactions in the browser; automated checks alone do not establish visual quality.

## Milestones and code review rules

A milestone is one coherent, commit-ready unit. Run relevant checks and self-review every change. Choose review depth by the complete task's behavior, blast radius, and failure consequences—not file/line counts or directory alone. Do not split risky work to avoid review.

| Review depth | When to use it |
| --- | --- |
| **Self-review; no subagents** | Local, low-risk edits with straightforward verification: copy, docs, styling, or an input minimum without shared validation, persistence, or API contract changes. |
| **One independent reviewer** | Substantive features, fixes, or refactors with bounded impact and no high-risk trigger. |
| **Three independent reviewers** | Behavior changes affecting authentication/authorization, account isolation, persisted-data compatibility/migrations, destructive operations, concurrency/resource lifecycles, shared API compatibility, inference numerical/device/memory behavior, or substantial architecture across owners. |

Review focuses: correctness/spec conformance; architecture/operational safety/performance/unnecessary complexity; test value/gaps/product quality, including UI accessibility. One reviewer covers all; three divide them. User instructions override defaults. Select the tier without asking permission; briefly explain spawning. Required checks and browser verification apply at every tier.

1. Review once per coherent candidate after checks. Supply the same fixed base, acceptance criteria, and complete task-owned diff (staged, unstaged, new files); identify unrelated changes. Use focused briefs and entry points, not full conversation inheritance.
2. Reviewers inspect independently, including necessary callers/contracts. They are read-only: no edits, commits, or recursive delegation. Findings need locations, impact, and failure scenarios or engineering costs; style preferences alone are not blockers.
3. Resolve all material findings, add useful regression coverage, and rerun affected checks. A clean review is final for an unchanged candidate. For subsequent edits, seek blocker-only follow-up from affected reviewers on changes and consequences. Broaden review and reassess the tier only when scope/risk materially changes.
4. Do not declare readiness with material findings or unexplained failing gates. Report resolved findings, passed/failed/unavailable checks, and remaining limitations. PRs omit subagent counts, tiers, and process narration.

If required independent review is unavailable, self-review the corresponding focuses and disclose this to the user; ordinary low-risk self-review needs no such disclaimer. Never label self-review independent. Implementation delegation is optional for independently useful tasks with clear ownership.

## Commands and environments

- Use **pnpm 10**, never npm/yarn, with package-owned lockfiles and `pnpm -C <package> ...`. Use `.nvmrc`'s Node version; consult package scripts, not root Makefile assumptions about webv2.
- Preserve the existing Python/accelerator environment and `uv.lock`. Fresh test setup: `uv sync --locked --extra test`; do not routinely replace accelerator backends.
- Root Python checks: `uv tool run ruff@0.11.2 check <paths>` and `uv tool run ruff@0.11.2 format --check <paths>`. Use 120-column formatting, absolute imports, and configured vendored-code exclusions.
- Focused tests: `uv run --no-sync pytest <test-paths>`; full CI: `uv run --no-sync pytest -n logical`. Hardware/quiet-machine timing tests use `-m slow` on development machines; excluded by default and CI. See `tests/AGENTS.md` for isolation rules.
- `-n logical` is sized for CI runners that own the machine. On a development box cap workers (`-n 4`, or plain `pytest`), set `OMP_NUM_THREADS` when a run must stay light, and do not start a suite while a generation is running.
- `uv run --no-sync mypy scripts/invokeai-web.py` uses current exclusions; it is an optional diagnostic, not a CI gate or comprehensive type coverage.
- Select checks by behavior; run required milestone gates before review. Repeat unchanged checks only for a reason.
- After test dependencies are available, install hooks per checkout: `uv run --no-sync pre-commit install`. Committing configuration does not install hooks.

## Scratch files and repository hygiene

- Never commit plans, work logs, investigation notes, review transcripts, or handoffs under any filename. Optional scratch belongs in `.scratch/agents/<task>/`; do not relabel it as durable docs.
- Version durable documentation of implemented architecture, decisions, APIs, and setup.
- Inspect staged diffs and new files for planning material. Ignore rules neither remove tracked files nor prevent force-adds. Never force-add planning artifacts or introduce `package-lock.json`/`yarn.lock`.

## Commits and pull requests

- Readiness does not authorize commits/pushes; follow existing user authorization without automatic per-subtask commits.
- Use Conventional Commits: short outcome-focused subject, then one to three factual lines of material changes without repetition or process narration.
- Use the configured Git author; no agent attribution or co-author trailers.
- Follow `.github/pull_request_template.md`: ordered headings, all applicable sections and inline guidance, omit only conditional sections, and tick only true checklist items.
