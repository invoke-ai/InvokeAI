# AGENTS.md

## Temporary Development Instructions
### Vector Layers & Splines Port to InvokeAI v7

These instructions are temporary and apply only to the initial development and manual testing phase of the Vector Layers & Splines port.

The primary goal is to establish working functionality in the v7 Canvas architecture with minimal development overhead.

## 1. Direct Implementation

- The main agent is explicitly authorized to inspect, create, modify, and delete project files directly.
- Do not spawn, delegate to, or request assistance from subagents.
- Do not perform mandatory independent reviews or delegate implementation tasks.
- Handle investigation, implementation, and debugging within the main agent session.
- Keep reasoning and progress reports concise to minimize unnecessary token consumption.

## 2. Automated Validation

Do not automatically run:

- Unit, integration, or end-to-end tests.
- TypeScript or Python type checks.
- ESLint, Ruff, or other linters.
- Formatting checks.
- Frontend or backend builds.
- Architecture or performance checks.
- Automated code reviews.

These operations may only be performed when explicitly requested by the user.

Do not run automated validation simply because a file was modified or an implementation milestone was reached.

## 3. Implementation Strategy

- Focus on functional compatibility with the InvokeAI v7 Canvas architecture.
- Use the existing v6 Vector Layers & Splines implementation as the behavioral reference: [`DustyShoe/InvokeAI`, branch `feat(canvas)--Splines-tool`](https://github.com/DustyShoe/InvokeAI/tree/feat%28canvas%29--Splines-tool).
- Use [the rough integration plan](.scratch/agents/splines-port/PLAN.md) as a starting point; adapt it to the current v7 architecture and implementation findings.
- Port functionality incrementally in logical, manageable stages.
- Preserve existing behavior wherever practical.
- Adapt implementation details when required by architectural differences between v6 and v7.
- Avoid unrelated refactoring, cleanup, or architectural changes.
- Do not introduce temporary workarounds without explaining their limitations.
- Prefer simple, maintainable solutions over unnecessary abstractions.

## 4. Manual Testing Workflow

The user is responsible for functional testing during this phase.

After completing a meaningful implementation milestone:

1. Stop making changes.
2. Briefly summarize what was implemented or modified.
3. Identify the functionality that requires manual verification.
4. Wait for user feedback before proceeding to the next milestone.

When the user reports a problem, investigate and fix it directly without automatically initiating automated validation or review.

## 5. Communication

- Keep responses concise and technically focused.
- Avoid repeating information already established in the conversation.
- Report actual implementation results rather than speculative success claims.
- Clearly identify incomplete functionality, known limitations, and unresolved issues.
- Do not generate lengthy implementation plans unless explicitly requested.
- Do not request confirmation for routine implementation details within an already approved task.

## 6. Finalization and Restoration

These temporary instructions must not remain in the final pull request.

When the user indicates that the port is functionally complete and ready for final validation:

1. Remind the user that this temporary `AGENTS.md` must be restored from `main`.
2. Recommend the following command:

   `git restore --source=main -- AGENTS.md`

3. Do not restore the file automatically without explicit user approval.
4. After restoration, follow the original repository instructions for final validation, testing, and review.
5. Ensure that temporary changes to `AGENTS.md` are excluded from the final pull request.

**Important:** Do not interpret ordinary implementation milestones as authorization to end temporary development mode. Only the user decides when the port is ready for finalization.

## 7. Instruction Priority

For this temporary porting phase, the rules in this file supersede the repository's normal delegation and automated-validation workflow wherever those rules conflict.

Higher-priority system and developer instructions remain applicable.

The temporary restrictions on subagents and automated validation remain in effect until the user explicitly authorizes a change or approves restoration of the original `AGENTS.md`.
