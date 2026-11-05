# Coordinating a task

For the main agent. Implementers and reviewers need only root `AGENTS.md`, the scoped files for their paths, and their brief.

## Delegation

- Main agent investigates, plans, briefs, integrates, validates, reports. It owns shared contracts and completion.
- Brief each implementer: objective, owned files, applicable `AGENTS.md`, agreed interfaces, acceptance criteria, focused checks. Implementer returns a short report: files changed, checks run, open issues.
- One writer per file at a time. Tightly coupled changes go to one implementer; split only independently verifiable work.
- Implementers own meaningful tests for their changes; by default never split production code from its tests.
- Follow-up fixes from review findings or failing checks go back to an implementer, preferably the original one.

## Review

User instructions override defaults. Pick the tier without asking; say briefly why you spawn reviewers. Required checks and browser verification apply at every tier.

1. Review once per coherent candidate, after checks and after every writer finishes. No edits to a candidate under review. Give all reviewers the same fixed base, acceptance criteria, full task-owned diff (staged, unstaged, new files). Name unrelated changes. Focused briefs and entry points, not full conversation history.
2. Put the reviewer's rules in the brief, so no reviewer has to find them:
   - Focuses: (1) correctness, spec conformance; (2) architecture, operational safety, performance, needless complexity; (3) test value and gaps, product quality, UI accessibility. One reviewer covers all three; three reviewers split them.
   - Inspect independently, including needed callers/contracts. Read-only: no edits, commits, or delegation.
   - Findings need location, impact, and failure scenario or engineering cost. Style preference alone is not a blocker.
3. Resolve all material findings. Add useful regression coverage. Rerun affected checks. Clean review is final for an unchanged candidate. Later edits: list changes since the reviewed candidate; get blocker-only follow-up from affected reviewers on them and their consequences. Broaden review or change tier only when scope/risk changes materially.
