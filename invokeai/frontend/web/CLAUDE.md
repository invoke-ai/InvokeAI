# Bash commands

All commands should be run from `<REPO_ROOT>/invokeai/frontend/web/`.

- `pnpm lint:prettier`: check formatting
- `pnpm lint:eslint`: check for linting issues
- `pnpm lint:knip`: check for unused dependencies
- `pnpm lint:dpdm`: check for dependency cycles
- `pnpm lint:tsc`: check for TypeScript issues
- `pnpm lint`: run all checks
- `pnpm fix`: automatically fix issues where possible
- `pnpm test:no-watch`: run the test suite

# Writing Tests

This repo uses `vitest` for unit tests.

Tests should be colocated with the code they test, and should use the `.test.ts` suffix.

Tests do not need to be written for code that is trivial or has no logic (e.g. simple type definitions, re-exports, etc.). We currently do not do UI tests.

# Agents

- Use @agent-javascript-pro and @agent-typescript-pro for JavaScript and TypeScript code generation and assistance.
- Use @frontend-developer for general frontend development tasks.

## Workflow

Split up tasks into smaller subtasks and handle them one at a time using an agent. Ensure each subtask is completed before moving on to the next.

Each agent should maintain a work log in a markdown file.

When an agent completes a task, it should:

1. Summarize the changes made.
2. List any files that were added, modified, or deleted.
3. Commit the changes with a descriptive commit message.

DO NOT PUSH ANY CHANGES TO THE REMOTE REPOSITORY.
