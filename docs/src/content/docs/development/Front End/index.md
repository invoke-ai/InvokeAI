---
title: Frontend Development
lastUpdated: 2026-02-18
---

Invoke's UI is made possible by many contributors and open-source libraries. Thank you!

## Dev environment

Follow the [dev environment](/development/setup/dev-environment/) guide to get set up. The default UI lives in `invokeai/frontend/webv2`. Run `make frontend-install`, then `make frontend-dev`; `make frontend-build` builds the bundle served by `invokeai-web`. The existing `frontendv2-*` targets remain aliases.

`invokeai/frontend/webv1` is the legacy frontend. Build it with `make frontend-legacy-build` and select it with `invokeai-web --web-legacy`. `--webv2` remains a compatibility alias for the default UI. Close other editor tabs before switching frontends on the same origin. Existing browser storage and project recovery data retain their names and formats; switching frontends does not migrate or erase them.

## Package scripts

Run these in `invokeai/frontend/webv2`:

- `dev`: run the frontend with hot reloading
- `build`: run formatting, lint, types, and architecture checks, then build
- `lint`: run formatting, Oxc lint, TypeScript, and architecture checks
- `fix`: fix supported lint and formatting issues
- `test`: run the unit suite
- `test:browser`: run Chromium interaction tests
- `check:release`: run the complete release gates, including performance, project-file, and accessibility journeys

The legacy package has its own scripts and lockfile. Frontend CI runs webv2's checks and release gate; legacy lint and tests remain available locally. See each package's `AGENTS.md` for its commands and ownership rules.

## Type generation

The shared `invokeai/frontend/api` package owns OpenAPI/type generation for backend contracts. CI checks these artifacts independently of either UI package. We use [openapi-typescript] to generate types from the app's OpenAPI schema. The generated types are committed to the repo in [schema.ts].

If you make backend changes, it's important to regenerate the frontend types:

```sh
set -o pipefail
pnpm -C invokeai/frontend/api install --frozen-lockfile
cd invokeai/frontend/api && python ../../../scripts/generate_openapi_schema.py | pnpm typegen
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
      "webRoot": "${workspaceFolder}/invokeai/frontend/webv2"
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

Follow `invokeai/frontend/webv2/AGENTS.md` and its `ARCHITECTURE.md` for ownership, state, React, persistence, and product-quality rules. The linked Redux and control-layer guides describe the legacy frontend.

## Commit format

Please use the [conventional commits] spec for the web UI, with a scope of "ui":

- `chore(ui): bump deps`
- `chore(ui): lint`
- `feat(ui): add some cool new feature`
- `fix(ui): fix some bug`

## Tests

Colocate unit tests and Chromium browser tests with the owning code. Use real browser storage and interaction where mocks cannot establish correctness. Run `pnpm check:release` before milestone readiness; browser screenshots and interaction review complement automated gates.

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
[schema.ts]: https://github.com/invoke-ai/InvokeAI-7/blob/main/invokeai/frontend/api/schema.ts
[conventional commits]: https://www.conventionalcommits.org/en/v1.0.0/
[Workflows - Design and Implementation]: ./workflows/
[State Management]: ./state-management/
