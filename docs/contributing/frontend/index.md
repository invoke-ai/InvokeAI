# Invoke UI

Invoke's UI is made possible by many contributors and open-source libraries. Thank you!

## Dev environment

Follow the [dev environment](../dev-environment.md) guide to get set up. Run the UI using `pnpm dev`.

## Package scripts

- `dev`: run the frontend in dev mode, enabling hot reloading
- `build`: run all checks (dpdm, eslint, prettier, tsc, knip) and then build the frontend
- `lint:dpdm`: check circular dependencies
- `lint:eslint`: check code quality
- `lint:prettier`: check code formatting
- `lint:tsc`: check type issues
- `lint:knip`: check for unused exports or objects
- `lint`: run all checks concurrently
- `fix`: run `eslint` and `prettier`, fixing fixable issues
- `test:ui`: run `vitest` with the fancy web UI

## Type generation

We use [openapi-typescript] to generate types from the app's OpenAPI schema. The generated types are committed to the repo in [schema.ts].

If you make backend changes, it's important to regenerate the frontend types:

```sh
cd invokeai/frontend/web && python ../../../scripts/generate_openapi_schema.py | pnpm typegen
```

On macOS and Linux, you can run `make frontend-typegen` as a shortcut for the above snippet.

## Localization

We use [i18next] for localization, but translation to languages other than English happens on our [Weblate] project.

Only the English source strings (i.e. `en.json`) should be changed on this repo.

## VSCode

### Example debugger config

```jsonc
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "chrome",
      "request": "launch",
      "name": "Invoke UI",
      "url": "http://localhost:5173",
      "webRoot": "${workspaceFolder}/invokeai/frontend/web"
    }
  ]
}
```

### Remote dev

We've noticed an intermittent timeout issue with the VSCode remote dev port forwarding.

We suggest disabling the editor's port forwarding feature and doing it manually via SSH:

```sh
ssh -L 9090:localhost:9090 -L 5173:localhost:5173 user@host
```

## Contributing Guidelines

Thanks for your interest in contributing to the Invoke Web UI!

Please follow these guidelines when contributing.

## Check in before investing your time

Please check in before you invest your time on anything besides a trivial fix, in case it conflicts with ongoing work or isn't aligned with the vision for the app.

If a feature request or issue doesn't already exist for the thing you want to work on, please create one.

Ping `@psychedelicious` on [discord] in the `#frontend-dev` channel or in the feature request / issue you want to work on - we're happy to chat.

## Code conventions

- This is a fairly complex app with a deep component tree. Please use memoization (`useCallback`, `useMemo`, `memo`) with enthusiasm.
- If you need to add some global, ephemeral state, please use [nanostores] if possible.
- Be careful with your redux selectors. If they need to be parameterized, consider creating them inside a `useMemo`.
- Feel free to use `lodash` (via `lodash-es`) to make the intent of your code clear.
- Please add comments describing the "why", not the "how" (unless it is really arcane).

## Commit format

Please use the [conventional commits] spec for the web UI, with a scope of "ui":

- `chore(ui): bump deps`
- `chore(ui): lint`
- `feat(ui): add some cool new feature`
- `fix(ui): fix some bug`

## Tests

We don't do any UI testing at this time, but consider adding tests for sensitive logic.

We use `vitest`, and tests should be next to the file they are testing. If the logic is in `something.ts`, the tests should be in `something.test.ts`.

In some situations, we may want to test types. For example, if you use `zod` to create a schema that should match a generated type, it's best to add a test to confirm that the types match. Use `tsafe`'s assert for this.

## Submitting a PR

- Ensure your branch is tidy. Use an interactive rebase to clean up the commit history and reword the commit messages if they are not descriptive.
- Run `pnpm lint`. Some issues are auto-fixable with `pnpm fix`.
- Fill out the PR form when creating the PR.
  - It doesn't need to be super detailed, but a screenshot or video is nice if you changed something visually.
  - If a section isn't relevant, delete it.

## Other docs

- [Workflows - Design and Implementation]
- [State Management]

[discord]: https://discord.gg/ZmtBAhwWhy
[i18next]: https://github.com/i18next/react-i18next
[Weblate]: https://hosted.weblate.org/engage/invokeai/
[openapi-typescript]: https://github.com/openapi-ts/openapi-typescript
[schema.ts]: https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/frontend/web/src/services/api/schema.ts
[conventional commits]: https://www.conventionalcommits.org/en/v1.0.0/
[Workflows - Design and Implementation]: ./workflows.md
[State Management]: ./state-management.md
